import os
import sys
import argparse
from dataclasses import dataclass
import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from torch.utils.data import TensorDataset, DataLoader

from utils import (
    set_random_seed,  make_loader, compute_class_weights,
    softmax_confidence, compute_source_prototypes, prototype_similarity,
    compute_classifier_margin, safe_GMM, percentile_rank,
    parse_ratio_schedule, cumulative_target_count, make_directory, make_logger, get_run_info,log_pseudo_selection
)
from model import Model
from data import  build_split_data


# ===================== Inference Helpers =====================

def encode_in_batches(model,x,device,batch_size):
    model.eval()
    dataset = TensorDataset(torch.from_numpy(x.astype(np.float32)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_feat, all_logits = [], []
    with torch.no_grad():
        for (bx,) in loader:  # TensorDataset contains only one tensor, so (bx,) is correct
            bx = bx.to(device)
            feat, logits = model(bx)
            all_feat.append(feat.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_feat), np.concatenate(all_logits)

def compute_raw_cosine_scores(model: Model, feats: np.ndarray) -> np.ndarray:
    # Extract classifier weights (w) and convert to float32
    w = model.classifier.weight.detach().cpu().numpy().astype(np.float32)
    # Inline normalization logic (feats)
    feats_norm = feats / np.clip(np.linalg.norm(feats, axis=1, keepdims=True), 1e-12, None)
    # Inline normalization logic (w)
    w_norm = w / np.clip(np.linalg.norm(w, axis=1, keepdims=True), 1e-12, None)
    # Compute cosine similarity (normalized features @ transpose of normalized weights)
    return (feats_norm @ w_norm.T).astype(np.float32)

# ===================== Pseudo-Label Selection =====================

@dataclass
class PseudoSelection:
    selected_mask: np.ndarray
    pseudo_labels: np.ndarray
    reliability: np.ndarray
    n_candidate: int
    n_selected: int

def select_pseudo_labels(model, x_src, y_src, x_remain, device, args,it, total_test_num, already_count, schedule,logger):
    '''
    x_src, y_src: all original training samples (excluding pseudo-labeled samples)
    x_remain: remaining test samples (not yet selected for pseudo-labeling)	shrinks each round
    already_count: number of samples already pseudo-labeled
    '''
    # ========== Part 1: Extract features ==========
    n_cls = int(y_src.max()) + 1
    src_feat, _ = encode_in_batches(model, x_src, device, args.batch_size)  #compute prototypes using all training samples
    tgt_feat, tgt_logits = encode_in_batches(model, x_remain, device, args.batch_size)  #compute representations for the remaining test samples
    # ========== Part 2: Evaluator 1 -- classifier scoring ==========
    clf_pred, clf_conf, _ = softmax_confidence(tgt_logits)
    raw_scores = compute_raw_cosine_scores(model, tgt_feat)
    clf_pred = raw_scores.argmax(1).astype(np.int64)  # override with raw cosine scores
    margin = compute_classifier_margin(raw_scores) #margin = top1 score - top2 score
    # ========== Part 3: Evaluator 2 -- prototype scoring ==========
    protos = compute_source_prototypes(src_feat, y_src, n_cls) #each row protos[c] is the normalized prototype vector for class c (norm = 1)
    proto_pred, proto_sim= prototype_similarity(tgt_feat, protos) #proto_pred[i] = 1 <- prototype predicts B cell | proto_sim[i] = 0.87 <- similarity to the nearest prototype
    # ========== Part 4: GMM separates high-/low-quality samples ==========
    prob_feat, feat_high = safe_GMM(values = proto_sim, higher_is_better=True,reg_covar=5e-4, seed= args.random_seed)
    prob_marg, marg_high = safe_GMM(values = margin, higher_is_better=True,reg_covar=5e-4, seed= args.random_seed)
    # ========== Part 5: Determine candidate eligibility ==========
    agreement = proto_pred == clf_pred #two independent methods (classifier uses softmax output; prototype uses nearest neighbor in feature space) predict the same class -> the prediction is likely correct [True, False, True, True, False, True] agree / disagree / agree / agree / disagree / agree
    candidate = feat_high & marg_high#a sample must satisfy both conditions: (1) sufficiently close to a prototype (high feature quality) and (2) sufficiently certain classifier prediction (large margin). Failing either condition disqualifies it.
    if args.require_prediction_agreement:
        candidate &= agreement
    # ========== Part 6: Ranking + combined score ==========
    rank_f = percentile_rank(proto_sim)
    rank_m = percentile_rank(margin)
    joint = np.minimum(rank_f, rank_m).astype(np.float32)
    reliability = np.minimum(joint, np.minimum(prob_feat, prob_marg)).astype(np.float32)
    # ========== Part 7: Calculate how many samples to select this round ==========
    target_cum = cumulative_target_count(total_test_num, it, schedule, args)
    needed = max(target_cum - already_count, 0)
    # ========== Part 8: Select top-N from candidates ==========
    selected = np.zeros(len(x_remain), dtype=bool)
    if needed > 0:
        idx = np.where(candidate)[0]  # use the variable directly instead of s["candidate"]
        if idx.size > 0:
            order = np.lexsort((
                -prob_marg[idx], -prob_feat[idx],
                -rank_m[idx], -rank_f[idx], -joint[idx],
            ))
            selected[idx[order[:min(needed, len(idx))]]] = True
    # ========== Part 9: Determine pseudo-label source ==========
    #If pseudo_label_source = "prototype", use prototype predictions (based on nearest neighbors in feature space) as pseudo labels
    #If set to "classifier", use classifier (softmax output) predictions as pseudo labels
    if args.pseudo_label_source == "prototype":
        pseudo = proto_pred.copy()
    else:
        pseudo = clf_pred.copy()
    # ========== Part 10: Summary + return ==========
    log_pseudo_selection(
        logger, candidate, selected, agreement,
        clf_conf, proto_sim, margin, reliability,
        needed, x_remain
    )
    return PseudoSelection(
        selected_mask=selected,
        pseudo_labels=pseudo,
        reliability=reliability,
        n_candidate=int(candidate.sum()),
        n_selected=int(selected.sum()),
    )


# ===================== Training =====================

def train_stage(model, train_data, train_label, data_pseudo, label_pseudo, reliability_pseudo, device, args,logger):
    if data_pseudo is not None and len(data_pseudo) > 0:
        # Pseudo labels available -> concatenate training data + pseudo data
        x = np.concatenate([train_data, data_pseudo]).astype(np.float32)
        y = np.concatenate([train_label, label_pseudo]).astype(np.int64)
        is_pseudo = np.concatenate([np.zeros(len(train_data)), np.ones(len(data_pseudo))]).astype(np.float32)#mark whether each sample is pseudo-labeled
        rel = np.concatenate([np.ones(len(train_data)), reliability_pseudo.astype(np.float32)]) #reliability of each sample	source samples are all 1.0; pseudo samples use model confidence
    else:
        # No pseudo labels (round 0) -> use source only
        x, y = train_data.astype(np.float32), train_label.astype(np.int64)
        is_pseudo = np.zeros(len(x), dtype=np.float32)
        rel = np.ones(len(x), dtype=np.float32)
    '''
    Index:    0    1    2   ...  999  1000  1001  ...  1199
    is_p:   0.0  0.0  0.0 ...  0.0   1.0   1.0  ...   1.0
    rel:    1.0  1.0  1.0 ...  1.0   0.95  0.87 ...   0.72
             <-- source (true labels) -->   <-- pseudo (pseudo labels) -->
    '''
    loader = make_loader(x, y, args.batch_size, shuffle=True, extra=[is_pseudo, rel])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    cw = compute_class_weights(train_label, int(train_label.max()) + 1, device, args.class_weight_mode)

    # ========== Early-stopping counter ==========
    above_threshold_count = 0  # number of consecutive epochs above the threshold
    early_stop_acc = 0.95
    early_stop_patience=1000000
    for ep in range(args.epoch_num):
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for bx, by, bp, br in loader:
            bx, by = bx.to(device), by.to(device)
            bp, br = bp.to(device), br.to(device)
            _, logits = model(bx)
            ce = F.cross_entropy(logits, by, weight=cw, reduction="none")
            pw = torch.full_like(bp, args.pseudo_loss_weight)
            if args.use_reliability_in_pseudo_weight:
                pw = pw * br
            sw = torch.where(bp > 0.5, pw, torch.ones_like(bp))
            loss = (ce * sw).sum() / sw.sum().clamp_min(1e-8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            total_correct += (logits.argmax(1) == by).sum().item()
            total_n += bx.shape[0]
        # ========== Calculate accuracy for this epoch ==========
        epoch_acc = total_correct / max(total_n, 1)
        epoch_loss = total_loss / max(len(loader), 1)
        logger.info(f"  Epoch [{ep}] "f"Loss: {epoch_loss:.4f}, "f"Training accuracy: {epoch_acc:.4f}")
        # ========== Early-stopping check ==========
        if epoch_acc >= early_stop_acc:
            above_threshold_count += 1
            if above_threshold_count >= early_stop_patience:
                logger.info(
                    f"  Early stopping triggered: {above_threshold_count} consecutive epochs "
                    f"with accuracy >= {early_stop_acc}; "
                    f"stopped at epoch {ep}"
                )
                break
        else:
            above_threshold_count = 0  # threshold not reached; reset counter to zero
    # ========== Training completion log ==========
    actual_epochs = ep + 1
    if above_threshold_count < early_stop_patience:
        logger.info(f"  Training for this round completed: ran all {actual_epochs} epochs (early stopping not triggered)")
    return actual_epochs  # return the actual number of training epochs (for log tracking)

# ===================== Evaluation =====================
def evaluate_final(model,x_src,y_src,x_test,y_test,device,args,logger,label_map,):
    # ========== Extract training-set and test-set embeddings together ==========
    n_cls = int(y_src.max()) + 1
    train_embedding, _ = encode_in_batches(model,x_src,device,args.batch_size)
    test_embedding, _ = encode_in_batches(model,x_test,device,args.batch_size)
    # Compute prototypes using only the filtered training set
    protos = compute_source_prototypes(train_embedding,y_src,n_cls,)
    def predict_from_embedding(feats):
        raw_scores = compute_raw_cosine_scores(model, feats)
        classifier_pred = raw_scores.argmax(1).astype(np.int64)
        prototype_pred, _ = prototype_similarity(feats, protos)
        if args.pseudo_label_source == "prototype":
            return prototype_pred.astype(np.int64)
        return classifier_pred.astype(np.int64)
    # ========== Predict the training set and test set separately ==========
    train_pred_int = predict_from_embedding(train_embedding)
    test_pred_int = predict_from_embedding(test_embedding)
    train_pred_labels = np.asarray([label_map[int(i)] for i in train_pred_int],dtype=object,)
    test_pred_labels = np.asarray([label_map[int(i)] for i in test_pred_int],dtype=object,)
    # ========== Test-set metrics ==========
    acc = float(accuracy_score(y_test, test_pred_int))
    macro_f1 = float(f1_score(y_test,test_pred_int,average="macro",zero_division=0))
    weighted_f1 = float(f1_score(y_test,test_pred_int,average="weighted",zero_division=0))
    kappa = float(cohen_kappa_score(y_test, test_pred_int))
    metrics = {"accuracy": acc,"macro_f1": macro_f1,"weighted_f1": weighted_f1,"kappa": kappa,}
    pred_source = ("Prototype"if args.pseudo_label_source == "prototype"else "Classifier")
    logger.info(f"  [Final evaluation] "f"Prediction source={pred_source}, "f"Training samples={len(x_src)}, "f"Test samples={len(y_test)}, "f"Accuracy={acc:.4f}, "f"Macro-F1={macro_f1:.4f}, "f"Weighted-F1={weighted_f1:.4f}, "f"Kappa={kappa:.4f}")

    return (metrics,train_embedding.astype(np.float32),test_embedding.astype(np.float32),train_pred_int.astype(np.int64),test_pred_int.astype(np.int64),train_pred_labels,test_pred_labels)

def save_prediction_h5ad(split,train_embedding,test_embedding,train_pred_labels,test_pred_labels,out_file,logger,
    source_batch="reference",target_batch="query",batch_key="Batch",):
    # ========== Basic checks ==========
    n_train = split.train_adata.n_obs
    n_test = split.test_adata.n_obs
    train_pred_labels = np.asarray(train_pred_labels,dtype=str,)
    test_pred_labels = np.asarray(test_pred_labels,dtype=str,)
    # ========== Combine embeddings ==========
    all_embedding = np.concatenate([train_embedding.astype(np.float32),test_embedding.astype(np.float32),],axis=0,)
    # ========== Build training-set obs ==========
    # Preserve all original obs information from the filtered training data
    train_obs = split.train_adata.obs.copy()
    train_obs[batch_key] = np.full(n_train,str(source_batch),dtype=object,)
    train_obs["predict"] = train_pred_labels
    # ========== Build test-set obs ==========
    # Preserve all original obs information from the filtered test data
    test_obs = split.test_adata.obs.copy()
    test_obs[batch_key] = np.full(n_test,str(target_batch),dtype=object,)
    test_obs["predict"] = test_pred_labels
    # Convert CellType and predict to strings consistently to avoid categorical merge issues
    if "CellType" in train_obs.columns:
        train_obs["CellType"] = train_obs["CellType"].astype(str)
    if "CellType" in test_obs.columns:
        test_obs["CellType"] = test_obs["CellType"].astype(str)
    train_obs[batch_key] = train_obs[batch_key].astype(str)
    test_obs[batch_key] = test_obs[batch_key].astype(str)
    train_obs["predict"] = train_obs["predict"].astype(str)
    test_obs["predict"] = test_obs["predict"].astype(str)
    # The order must match all_embedding:
    # training samples first, test samples second
    all_obs = pd.concat([train_obs, test_obs],axis=0,sort=False,)
    if all_obs.index.has_duplicates:
        logger.warning("Duplicate cell names exist in the output data. ""The file will still be saved, but take care when indexing by obs_names later.")
    # ========== Embedding dimension information ==========
    latent_dim = all_embedding.shape[1]
    var = pd.DataFrame(index=[f"latent_{i}" for i in range(latent_dim)])
    # ========== Create AnnData ==========
    out_adata = ad.AnnData(X=all_embedding,obs=all_obs,var=var,)
    out_adata.write_h5ad(out_file,compression="gzip",)
    logger.info(f"Prediction h5ad saved: {out_file}")



# ===================== Main =====================

def main(args):
    # ---- Build output path and create directory ----
    args.output_dir = os.path.join(args.o, args.o_name)
    make_directory(args.output_dir)
    log_file = os.path.join(args.output_dir, "run.log")
    logger = make_logger(title="CellType", filename=log_file, level="INFO", trace=False)
    logger.info(get_run_info(sys.argv, args)) # record run information (command-line arguments, etc.)

    set_random_seed(args.random_seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    split = build_split_data(args,logger)

    cell_class_num =  len(split.valid_labels)
    feature_dim = split.train_x.shape[1]

    schedule = parse_ratio_schedule(args.pseudo_ratio_schedule, args.max_iteration)
    logger.info(f"Schedule: {schedule}")
    logger.info("======= Training Start =======")

    data_pseudo_samples = None
    label_pseudo_samples  = None
    reliability_pseudo_samples = None

    remaining = split.test_x.copy()  #test-set samples not yet selected for pseudo-labeling; decreases each round
    total_test_num  = remaining.shape[0]
    model = None
    last_selection = None

    for it in range(args.max_iteration + 1):
        # Initialize or reinitialize model
        if model is None or not args.keep_weights_each_iteration:
            model = Model(feature_dim, cell_class_num, args).to(device)
        logger.info(f"--- Round: {it} | Pseudo-label count={0 if data_pseudo_samples is None else len(data_pseudo_samples)} ---")
        actual_ep = train_stage(
            model, split.train_x, split.train_y,
            data_pseudo_samples, label_pseudo_samples, reliability_pseudo_samples,
            device, args, logger)

        if it == args.max_iteration or remaining.shape[0] == 0:
            break

        # Select pseudo labels
        sel = select_pseudo_labels(model, split.train_x, split.train_y, remaining, device, args,
            it, total_test_num, 0 if data_pseudo_samples is None else len(data_pseudo_samples), schedule,logger
        )
        if sel.n_selected == 0:
            logger.info("  No pseudo labels selected. Stopping.")
            break

        # Accumulate  sel.selected_mask is a boolean array of length len(remaining); selected positions are True
        sx = remaining[sel.selected_mask] #input features of samples selected in this round, i.e. the matrix
        sy = sel.pseudo_labels[sel.selected_mask]#pseudo labels of samples selected in this round (from the classifier or prototype)
        sr = sel.reliability[sel.selected_mask]#reliability of samples selected in this round (source of training weights)
        if data_pseudo_samples is None:# first pseudo-label selection; assign directly
            data_pseudo_samples, label_pseudo_samples, reliability_pseudo_samples = sx.copy(), sy.copy(), sr.copy()
        else:# subsequent rounds; concatenate with existing pseudo labels
            data_pseudo_samples = np.concatenate([data_pseudo_samples, sx])
            label_pseudo_samples = np.concatenate([label_pseudo_samples, sy])
            reliability_pseudo_samples = np.concatenate([reliability_pseudo_samples, sr])
        remaining = remaining[~sel.selected_mask]
    logger.info("======= Training Done =======")

    # Final evaluation
    (
        metrics,
        train_embedding,
        test_embedding,
        train_pred_int,
        test_pred_int,
        train_pred_label,
        test_pred_label,
    ) = evaluate_final(model,split.train_x,split.train_y,split.test_x,split.test_y,device,args,logger,label_map=split.label_map)

    pred_h5ad_file = os.path.join(args.output_dir, 'embedding.h5ad')

    save_prediction_h5ad(
        split=split,
        train_embedding=train_embedding,
        test_embedding=test_embedding,
        train_pred_labels=train_pred_label,
        test_pred_labels=test_pred_label,
        out_file=pred_h5ad_file,
        logger=logger,
        source_batch=args.source_batch,
        target_batch=args.target_batch,
        batch_key="Batch",
    )

# ===================== CLI =====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('-i', "--input_h5ad", type=str, required=True)
    parser.add_argument("--source_batch", type=str, default='reference')
    parser.add_argument("--target_batch", type=str, default='query')
    parser.add_argument("--del_unknown",action='store_true', help='remove unknown cell type', default=False)
    parser.add_argument("--del_rare", action='store_true', help='remove unknown cell type', default=False)

    # Model / optimization
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_iteration", type=int, default=5)
    parser.add_argument("--epoch_num", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--random_seed", type=int, default=2026)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--resmlp_blocks", type=int, default=3)
    parser.add_argument("--resmlp_expansion", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cosine_scale", type=float, default=20)
    parser.add_argument("--learnable_scale", action="store_true")

    # Pseudo-label selection
    parser.add_argument("--pseudo_label_source", type=str, default="classifier", choices=["prototype", "classifier"])
    parser.add_argument("--pseudo_ratio_schedule", type=str, default="0.10,0.30,0.50,0.70")
    parser.add_argument("--first_round_min_count", type=int, default=100)
    parser.add_argument("--first_round_max_count", type=int, default=400)
    parser.add_argument("--max_total_pseudo_count", type=int, default=99999999)
    parser.set_defaults(require_prediction_agreement=True)

    # Loss
    parser.add_argument("--class_weight_mode", type=str, default="none", choices=["none", "inv", "sqrt_inv"])
    parser.add_argument("--pseudo_loss_weight", type=float, default=0.3)
    parser.add_argument("--use_reliability_in_pseudo_weight", action="store_true", default=True)
    parser.add_argument("--keep_weights_each_iteration", action="store_true")

    # Output
    parser.add_argument("--o", type=str, required=True,
                        help="Base output path, e.g. /home/user/results/")
    parser.add_argument("--o_name", type=str, required=True,
                        help="Output folder name, e.g. experiment_01")

    main(parser.parse_args())

