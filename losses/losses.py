import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

from models.HoVerNet.utils import mse_loss, msge_loss


def _student_loss(pred_student, true):
    """
    Weighted hovernt loss
    :param pred_student:
    :param true:
    :return:
    """
    WEIGHTS_GT = [0.10654577579627722,
                  0.3310121522330427,
                  0.7301252761147444,
                  0.5030386955824504,
                  3.711333187208553,
                  0.6179449130649323]
    WEIGHTS_BINARY_GT = [0.6186044463779635, 1.3813955536220366]
    BINARY_WEIGHTS = torch.from_numpy(np.array(WEIGHTS_BINARY_GT).astype('float32')).cuda()

    WEIGHTS_GT = torch.from_numpy(np.array(WEIGHTS_GT).astype('float32')).cuda()
    student_np, student_hv, student_tp = pred_student
    true_np, true_hv, true_tp = true
    true_tp = torch.argmax(true_tp, dim=1)
    dice_loss_b = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True)
    dice_loss_c = smp.losses.DiceLoss(mode=smp.losses.MULTICLASS_MODE, from_logits=True)
    la = mse_loss(true_hv, student_hv)
    lb = msge_loss(true_hv, student_hv, true_np)
    lc = dice_loss_b(student_np, true_np)
    ld = F.cross_entropy(student_np, true_np, weight=BINARY_WEIGHTS)
    le = dice_loss_c(student_tp, true_tp)
    lf = F.cross_entropy(student_tp, true_tp, weight=WEIGHTS_GT)
    ca = 1
    cb = 1
    cc = 1
    cd = 1
    ce = 1
    cf = 1
    hv_loss = ca * la + cb * lb + cc * lc + cd * ld + ce * le + cf * lf
    return hv_loss


def _distill_loss(pred_student, pred_teacher, T):
    WEIGHTS_HOVERNET = [0.12137613059784041,
                        0.3585983027106687,
                        0.962839423013976,
                        0.6344337609164922,
                        3.1918593602633436,
                        0.730893022497679]

    WEIGHTS_BINARY_HOVERNET = [0.6055308830505653, 1.3944691169494348]
    student_np, student_hv, student_tp = pred_student
    teacher_np, teacher_hv, teacher_tp = pred_teacher
    BINARY_WEIGHTS = torch.from_numpy(
        np.array(WEIGHTS_BINARY_HOVERNET).astype('float32')
    ).cuda()
    WEIGHTS_HOVERNET = torch.from_numpy(np.array(WEIGHTS_HOVERNET).astype('float32')).cuda()

    cewb = F.cross_entropy(student_np, F.softmax(teacher_np, dim=1), weight=BINARY_WEIGHTS)
    cewm = F.cross_entropy(student_tp, F.softmax(teacher_tp, dim=1), weight=WEIGHTS_HOVERNET)

    KD_loss_np = F.kl_div(F.log_softmax(student_np / T, dim=1),
                          F.softmax(teacher_np / T, dim=1), reduction='mean') * (T * T)
    KD_loss_tp = F.kl_div(F.log_softmax(student_tp / T, dim=1),
                          F.softmax(teacher_tp / T, dim=1), reduction='mean') * (T * T)

    mse_loss_hv = mse_loss(teacher_hv, student_hv)
    msge_loss_hv = msge_loss(teacher_hv, student_hv, F.softmax(teacher_np, dim=1)[:, 1, ...])

    distill_loss = KD_loss_np + cewb + KD_loss_tp + cewm + mse_loss_hv + msge_loss_hv

    return distill_loss


def loss_fcn(pred_student, pred_teacher, true, alpha, T):
    student_np, student_hv, student_tp = pred_student[:, :2, ...], pred_student[:, 2:4, ...], pred_student[:, 4:, ...]
    student_hv = torch.permute(student_hv, (0, 2, 3, 1))
    loss_distill, loss_student = None, None
    if pred_teacher is not None:
        teacher_np, teacher_hv, teacher_tp = pred_teacher[:, :2, ...], pred_teacher[:, 2:4, ...], pred_teacher[:, 4:,
                                                                                                  ...]
        teacher_hv = torch.permute(teacher_hv, (0, 2, 3, 1))
        loss_distill = _distill_loss((student_np, student_hv, student_tp),
                                     (teacher_np, teacher_hv, teacher_tp), T)

    if true is not None:
        true_np, true_hv, true_tp = true[:, 0, ...].type(torch.long), true[:, 1:3, ...], true[:, 3:, ...]
        true_hv = torch.permute(true_hv, (0, 2, 3, 1))
        loss_student = _student_loss((student_np, student_hv, student_tp),
                                     (true_np, true_hv, true_tp))

    if pred_teacher is not None and true is not None:
        loss = alpha * loss_distill + (1 - alpha) * loss_student
    elif pred_teacher is None and true is not None:
        loss = loss_student
    elif pred_teacher is not None and true is None:
        loss = loss_distill
    else:
        raise Exception()

    return loss
