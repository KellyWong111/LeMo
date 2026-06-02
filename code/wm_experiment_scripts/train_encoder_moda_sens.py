from functools import partial

import torch

import train_encoder_moda as base


def lejepa_forward_sens(self, batch, stage, cfg):
    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]
    tgt_emb = emb[:, n_preds:]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    sens_cfg = cfg.loss.get("action_sens", {})
    sens_enabled = bool(sens_cfg.get("enabled", False))
    sens_loss = pred_emb.new_tensor(0.0)
    if sens_enabled:
        sigma = float(sens_cfg.get("sigma", 0.15))
        tau = float(sens_cfg.get("tau", 0.25))
        eps = float(sens_cfg.get("eps", 1e-6))

        noise = torch.randn_like(batch["action"][:, :ctx_len]) * sigma
        perturbed_action = batch["action"][:, :ctx_len] + noise
        perturbed_act_emb = self.model.action_encoder(perturbed_action)
        perturbed_pred = self.model.predict(ctx_emb, perturbed_act_emb)

        dz = (perturbed_pred[:, -1] - pred_emb[:, -1].detach()).norm(dim=-1)
        da = noise.reshape(noise.shape[0], -1).norm(dim=-1).clamp_min(eps)
        ratio = dz / da
        sens_loss = torch.relu(tau - ratio).mean()
        output["action_sens_loss"] = sens_loss
        output["action_sens_ratio"] = ratio.mean().detach()

    weight = float(sens_cfg.get("weight", 0.0)) if sens_enabled else 0.0
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"] + weight * sens_loss

    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
    if sens_enabled:
        losses_dict[f"{stage}/action_sens_ratio"] = output["action_sens_ratio"]
    self.log_dict(losses_dict, on_step=True, sync_dist=True)
    return output


base.lejepa_forward = lejepa_forward_sens


if __name__ == "__main__":
    base.run()
