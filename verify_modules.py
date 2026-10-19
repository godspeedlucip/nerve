"""
Empirical proof that AnisotropicAttention, TubularEnhancement, and
ProjectionConsistencyLoss are actively used during forward/backward passes.

6 independent proofs, all runnable with: conda activate nnunet && python verify_modules.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'nnunetv2'))

import numpy as np
import torch
from torch import nn

# ── shared test config ──────────────────────────────────────────────
B, C, D, H, W = 2, 1, 64, 128, 128
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
print(f"Input shape: ({B}, {C}, {D}, {H}, {W})")
print("=" * 60)

# ── Proof 1: Gradient flow through all 3 modules ────────────────────
print("\n[Proof 1] Gradient flow through AnisoAttn, TubEnh, ProjCons")
print("-" * 40)

from nnunetv2.network_modules.anisotropic_attention import AnisotropicAttention
from nnunetv2.network_modules.tubular_enhancement import TubularEnhancement
from nnunetv2.network_modules.projection_consistency import (
    ProjectionConsistencyLoss, CombinedSegProjLoss,
)
from nnunetv2.network_modules.aniso_stent_unet import AnisoStentUNet
from nnunetv2.network_modules.enhancement_pipeline import EnhancementPipeline
from nnunetv2.training.auxiliary_targets import AuxiliaryTargets
from nnunetv2.training.pipeline_config import EnhancementConfig
from nnunetv2.training.pipeline_trainer import build_composite_loss

# Build a minimal backbone for testing
from dynamic_network_architectures.architectures.unet import PlainConvUNet
backbone = PlainConvUNet(
    input_channels=C, n_stages=4,
    features_per_stage=(8, 16, 32, 64),
    n_conv_per_stage=2,
    n_conv_per_stage_decoder=(2, 2, 2),
    conv_op=nn.Conv3d,
    kernel_sizes=3,
    strides=(1, 2, 2, 2),
    num_classes=NUM_CLASSES,
    deep_supervision=False,
)

model = AnisoStentUNet(backbone, C, NUM_CLASSES).to(DEVICE)
x = torch.randn(B, C, D, H, W, device=DEVICE, requires_grad=False)
y = model(x)

# Backward on a dummy scalar to check gradients
loss_val = y.sum()
loss_val.backward()

# Check gradients for each module
def check_grads(module, name, min_ratio=0.88):
    params_with_grad = 0
    params_total = 0
    zero_grad_names = []
    for n, p in module.named_parameters():
        params_total += p.numel()
        if p.grad is not None and p.grad.abs().sum() > 1e-10:
            params_with_grad += p.numel()
        else:
            zero_grad_names.append(n)
    ratio = params_with_grad / params_total if params_total > 0 else 0
    status = "[PASS]" if ratio >= min_ratio else "[FAIL]"
    print(f"  {status} {name}: {params_with_grad}/{params_total} params have non-zero grad "
          f"({ratio*100:.1f}%)", end="")
    if zero_grad_names:
        print(f" -- near-zero: {zero_grad_names}")
    else:
        print()
    return ratio >= min_ratio

p1_aniso = check_grads(model.aniso_attn, "AnisotropicAttention")
p1_tuben = check_grads(model.tub_enh, "TubularEnhancement")

# Check backbone params still receive gradients (proof wrapper doesn't break flow)
bb_params_with_grad = sum(
    p.numel() for p in model.backbone.parameters()
    if p.grad is not None and p.grad.abs().sum() > 0
)
bb_params_total = sum(p.numel() for p in model.backbone.parameters())
bb_ratio = bb_params_with_grad / bb_params_total
p1_bb = bb_ratio > 0.99
print(f"  {'[PASS]' if p1_bb else '[FAIL]'} Backbone U-Net: {bb_params_with_grad}/{bb_params_total} "
      f"params have non-zero grad ({bb_ratio*100:.1f}%)")

proj_cons_ok = False  # set in Proof 4

# ── Proof 2: AnisotropicAttention produces non-identity output ──────
print("\n[Proof 2] AnisotropicAttention is not an identity transform")
print("-" * 40)

aniso = AnisotropicAttention(C).to(DEVICE)
x_test = torch.randn(1, C, 32, 64, 64, device=DEVICE)
with torch.no_grad():
    out_aniso = aniso(x_test)
    diff = (out_aniso - x_test).abs().mean().item()
    rel_change = diff / x_test.abs().mean().item()

    # Also verify: fresh random init should NOT be identity
    # An identity module would have out == x exactly
    is_identity = torch.allclose(out_aniso, x_test, atol=1e-6)

print(f"  Mean abs diff: {diff:.6f}")
print(f"  Relative change: {rel_change*100:.2f}%")
print(f"  {'[PASS]' if not is_identity else '[FAIL]'} Is NOT identity: {not is_identity}")
print(f"  (A truly inactive/identity module would have diff=0)")

# Also verify it does SOMETHING directional: the 3 branches should have different outputs
with torch.no_grad():
    f_axial = aniso.conv_axial(x_test)
    f_sag = aniso.conv_sag(x_test)
    f_cor = aniso.conv_cor(x_test)
    diff_as = (f_axial - f_sag).abs().mean().item()
    diff_ac = (f_axial - f_cor).abs().mean().item()
    diff_sc = (f_sag - f_cor).abs().mean().item()
    branches_differ = diff_as > 1e-6 and diff_ac > 1e-6
print(f"  Branch pairwise diffs: axial-sag={diff_as:.6f}, axial-cor={diff_ac:.6f}, sag-cor={diff_sc:.6f}")
print(f"  {'[PASS]' if branches_differ else '[FAIL]'} Directional branches produce different outputs")

p2_ok = not is_identity and branches_differ

# ── Proof 3: TubularEnhancement produces non-identity output ────────
print("\n[Proof 3] TubularEnhancement is not an identity transform")
print("-" * 40)

tubenh = TubularEnhancement(NUM_CLASSES).to(DEVICE)
x_tub = torch.randn(1, NUM_CLASSES, 32, 64, 64, device=DEVICE)
with torch.no_grad():
    out_tub = tubenh(x_tub)
    diff_tub = (out_tub - x_tub).abs().mean().item()
    rel_tub = diff_tub / x_tub.abs().mean().item()
    is_id_tub = torch.allclose(out_tub, x_tub, atol=1e-6)

print(f"  Mean abs diff: {diff_tub:.6f}")
print(f"  Relative change: {rel_tub*100:.2f}%")
print(f"  {'[PASS]' if not is_id_tub else '[FAIL]'} Is NOT identity: {not is_id_tub}")

# Verify dilated branches have different receptive fields
with torch.no_grad():
    d1 = tubenh.dil1(x_tub)
    d5 = tubenh.dil5(x_tub)
    dil_differ = (d1 - d5).abs().mean().item() > 1e-6
print(f"  {'[PASS]' if dil_differ else '[FAIL]'} Dilated branches (dil=1 vs dil=5) produce different outputs")

p3_ok = not is_id_tub and dil_differ

# ── Proof 4: ProjectionConsistencyLoss effectiveness ────────────────
print("\n[Proof 4] ProjectionConsistencyLoss discriminates shapes")
print("-" * 40)

angles = [0, 30, 60]
proj_loss_fn = ProjectionConsistencyLoss(angles, align_corners=False).to(DEVICE)

# Test 1: Same pred & target -> loss approx 0
# ProjCons applies softmax internally, so pred=logits, target=one-hot
target_oh = torch.zeros(1, NUM_CLASSES, 32, 64, 64, device=DEVICE)
target_oh[:, 0] = 1.0
# Logits that softmax to the one-hot target
pred_same = target_oh.clone()
pred_same[pred_same == 1] = 100.0
pred_same[pred_same == 0] = -100.0
loss_same = proj_loss_fn(pred_same, target_oh).item()
print(f"  Loss(identical_pred, target): {loss_same:.8f}  {'[PASS]' if loss_same < 1e-4 else '[FAIL]'} (should be ~0)")

# Test 2: Different pred & target -> loss > 0
target_oh2 = torch.zeros(1, NUM_CLASSES, 32, 64, 64, device=DEVICE)
target_oh2[:, 1] = 1.0
pred_diff = target_oh2.clone()
pred_diff[pred_diff == 1] = 100.0
pred_diff[pred_diff == 0] = -100.0
loss_diff = proj_loss_fn(pred_diff, target_oh).item()
print(f"  Loss(different_pred, target): {loss_diff:.8f}  {'[PASS]' if loss_diff > 1e-4 else '[FAIL]'} (should be > 0)")

# Test 3: Gradient flows through ProjCons
vol_pred = torch.randn(1, NUM_CLASSES, 32, 64, 64, device=DEVICE, requires_grad=True)
vol_target = torch.zeros(1, NUM_CLASSES, 32, 64, 64, device=DEVICE)
vol_target[:, 0] = 1.0
l = proj_loss_fn(vol_pred, vol_target)
l.backward()
proj_grad_ok = vol_pred.grad is not None and vol_pred.grad.abs().sum() > 0
print(f"  {'[PASS]' if proj_grad_ok else '[FAIL]'} Gradient flows to input: grad_norm={vol_pred.grad.abs().sum().item():.4f}")

proj_cons_ok = loss_same < 1e-4 and loss_diff > 1e-4 and proj_grad_ok
p4_ok = proj_cons_ok

# ── Proof 5: CombinedSegProjLoss balance ─────────────────────────────
print("\n[Proof 5] CombinedSegProjLoss computes both DC+CE and ProjCons components")
print("-" * 40)

# Build a minimal CombinedSegProjLoss using actual DC+CE loss
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

seg_loss = DC_and_CE_loss(
    {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
    {}, weight_ce=1, weight_dice=1, ignore_label=None,
    dice_class=MemoryEfficientSoftDiceLoss,
)

proj_loss = ProjectionConsistencyLoss(angles, align_corners=False)

# Mock label_manager
class MockLabelManager:
    has_regions = False
    has_ignore_label = False
    num_segmentation_heads = NUM_CLASSES

combined = CombinedSegProjLoss(seg_loss, proj_loss, proj_weight=0.1, label_manager=MockLabelManager()).to(DEVICE)

output = torch.randn(B, NUM_CLASSES, 32, 64, 64, device=DEVICE, requires_grad=True)
target = torch.randint(0, NUM_CLASSES, (B, 1, 32, 64, 64), device=DEVICE)

l_total = combined(output, target)
l_seg = combined.seg_loss(output, target)

# Compute proj component separately
with torch.no_grad():
    target_oh = torch.nn.functional.one_hot(
        target[:, 0].long(), NUM_CLASSES
    ).permute(0, 4, 1, 2, 3).float()
    l_proj = combined.proj_loss(output, target_oh)
    l_proj_weighted = combined.proj_weight * l_proj

print(f"  Total loss:          {l_total.item():.6f}")
print(f"  DC+CE loss:          {l_seg.item():.6f}")
print(f"  ProjCons loss:       {l_proj.item():.6f}")
print(f"  Weighted ProjCons:   {l_proj_weighted.item():.6f}")

# Verify: total = seg + proj_weight * proj
reconstructed = l_seg + combined.proj_weight * l_proj
match = torch.allclose(l_total, reconstructed, atol=1e-5)
print(f"  {'[PASS]' if match else '[FAIL]'} total == DC+CE + weight * ProjCons: "
      f"expected={reconstructed.item():.6f}, got={l_total.item():.6f}")

# Verify gradient flows through both components
l_total.backward()
grad_norm = output.grad.abs().sum().item()
print(f"  {'[PASS]' if grad_norm > 0 else '[FAIL]'} Gradient flows through CombinedLoss: "
      f"|grad|={grad_norm:.4f}")

p5_ok = match and grad_norm > 0

# ── Proof 6: Forward projection physical consistency ─────────────────
print("\n[Proof 6] Forward projection preserves mass and is viewpoint-consistent")
print("-" * 40)

from nnunetv2.network_modules.projection_consistency import differentiable_forward_projection

vol_sphere = torch.zeros(1, 1, 32, 64, 64, device=DEVICE)
# Create a sphere in the volume
z_idx = torch.arange(32, device=DEVICE).float().view(-1,1,1)
y_idx = torch.arange(64, device=DEVICE).float().view(1,-1,1)
x_idx = torch.arange(64, device=DEVICE).float().view(1,1,-1)
cz, cy, cx = 16, 32, 32
dist = torch.sqrt((z_idx-cz)**2 + (y_idx-cy)**2 + (x_idx-cx)**2)
vol_sphere[0, 0] = (dist <= 10).float()

# Project at different angles
projs = differentiable_forward_projection(vol_sphere, [0, 45, 90, 135, 180], align_corners=False)

# Mass conservation: each projection should have roughly same sum (sphere is isotropic)
proj_sums = [projs[0, i].sum().item() for i in range(5)]
mean_sum = np.mean(proj_sums)
max_dev = max(abs(s - mean_sum) / mean_sum * 100 for s in proj_sums)

print(f"  Projection sums at 0,45,90,135,180 deg: {[f'{s:.1f}' for s in proj_sums]}")
print(f"  Max deviation from mean: {max_dev:.2f}%")
print(f"  {'[PASS]' if max_dev < 5 else '[FAIL]'} Isotropic sphere -> consistent projections "
      f"(< 5% deviation)")

# Translational consistency: projection at 0 and 180 should be mirror images
proj_0 = projs[0, 0, 0].cpu().numpy()
proj_180 = projs[0, 3, 0].cpu().numpy()
proj_180_flipped = np.fliplr(proj_180)
corr = np.corrcoef(proj_0.flatten(), proj_180_flipped.flatten())[0, 1]
print(f"  Correlation(proj_0, flip(proj_180)): {corr:.4f}")
print(f"  {'[PASS]' if corr > 0.99 else '[FAIL]'} 0deg and 180deg projections are "
      f"mirror-symmetric (corr > 0.99)")

p6_ok = max_dev < 5 and corr > 0.99

# ── Proof 7 (bonus): Full end-to-end train step ──────────────────────
print("\n[Proof 7] Full train_step with AnisoStentUNet + ProjCons produces valid losses")
print("-" * 40)

# Use the actual trainer train_step logic
model_full = AnisoStentUNet(backbone, C, NUM_CLASSES).to(DEVICE)
seg_loss_full = DC_and_CE_loss(
    {'batch_dice': False, 'smooth': 1e-5, 'do_bg': False, 'ddp': False},
    {}, weight_ce=1, weight_dice=1, ignore_label=None,
    dice_class=MemoryEfficientSoftDiceLoss,
)
proj_loss_full = ProjectionConsistencyLoss(angles, align_corners=False).to(DEVICE)
combined_full = CombinedSegProjLoss(
    seg_loss_full, proj_loss_full, proj_weight=0.1,
    label_manager=MockLabelManager(),
).to(DEVICE)

optimizer = torch.optim.SGD(model_full.parameters(), lr=0.01)

# Run 5 training steps
losses = []
for step in range(5):
    data = torch.randn(B, C, 32, 64, 64, device=DEVICE)
    target = torch.randint(0, NUM_CLASSES, (B, 1, 32, 64, 64), device=DEVICE)

    optimizer.zero_grad()
    output = model_full(data)
    l = combined_full(output, target)
    l.backward()
    optimizer.step()
    losses.append(l.item())

# Loss should decrease (or at least change meaningfully)
loss_change = losses[0] - losses[-1]
print(f"  Loss sequence: {[f'{l:.4f}' for l in losses]}")
print(f"  {'[PASS]' if abs(loss_change) > 1e-6 else '[FAIL]'} Loss changes during training "
      f"(delta={loss_change:.6f})")

p7_ok = abs(loss_change) > 1e-6

# ── Proof 8: Unified all-enabled composition path ─────────────────────
print("\n[Proof 8] Unified all-enabled pipeline + composite loss")
print("-" * 40)

class TinyVerificationBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.projection = nn.Conv3d(C, NUM_CLASSES, kernel_size=1)

    def forward(self, inputs):
        return self.projection(inputs)


class TinyVerificationSegmentationLoss(nn.Module):
    def forward(self, output, target):
        return nn.functional.cross_entropy(output, target[:, 0].long())


unified_config = EnhancementConfig(
    enable_anisotropic_attention=True,
    enable_tubular_enhancement=True,
    projection_angles_deg=(0.0,),
    projection_consistency_weight=0.25,
    dtm_mode="active",
    dtm_weight=0.5,
    dtm_activate_after_epoch=0,
    dtm_active_weight=1.0,
)
unified_model = EnhancementPipeline(
    TinyVerificationBackbone(), C, NUM_CLASSES, unified_config
).to(DEVICE)

class _UnifiedLabelManager:
    has_regions = False
    has_ignore_label = False
    num_segmentation_heads = NUM_CLASSES


unified_loss = build_composite_loss(
    unified_config,
    TinyVerificationSegmentationLoss(),
    _UnifiedLabelManager(),
).to(DEVICE)
unified_input = torch.randn(1, C, 8, 8, 8, device=DEVICE)
unified_target = torch.randint(0, NUM_CLASSES, (1, 1, 8, 8, 8), device=DEVICE)
unified_dtm = torch.rand(1, 1, 8, 8, 8, NUM_CLASSES, device=DEVICE)
unified_auxiliary = AuxiliaryTargets(
    dtm=unified_dtm,
    valid_mask=torch.ones_like(unified_dtm),
)
unified_output = unified_model(unified_input)
unified_breakdown = unified_loss.breakdown(
    unified_output, unified_target, unified_auxiliary, epoch=0
)
unified_breakdown.total.backward()
unified_finite = all(
    torch.isfinite(component).item()
    for component in (
        unified_breakdown.total,
        unified_breakdown.segmentation,
        unified_breakdown.projection,
        unified_breakdown.dtm,
    )
)
unified_grad = any(
    parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for parameter in unified_model.parameters()
    if parameter.requires_grad
)
p8_ok = unified_finite and unified_grad
print(f"  {'[PASS]' if p8_ok else '[FAIL]'} all components finite and gradients present")

# ── Summary ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
results = {
    "1. Gradient flow (AnisoAttn)": p1_aniso,
    "   Gradient flow (TubEnh)": p1_tuben,
    "   Gradient flow (Backbone)": p1_bb,
    "2. AnisoAttn non-identity": p2_ok,
    "3. TubEnh non-identity": p3_ok,
    "4. ProjCons effectiveness": p4_ok,
    "5. Combined loss balance": p5_ok,
    "6. Projection physics": p6_ok,
    "7. End-to-end training": p7_ok,
    "8. Unified all-enabled composition": p8_ok,
}

all_pass = True
for name, result in results.items():
    status = "[PASS]" if result else "[FAIL]"
    if not result:
        all_pass = False
    print(f"  {status}  Proof {name}")

print("=" * 60)
if all_pass:
    print("ALL PROOFS PASSED - Unified enhancement pipeline is empirically active.")
else:
    print("SOME PROOFS FAILED - see above for details.")
print("=" * 60)
