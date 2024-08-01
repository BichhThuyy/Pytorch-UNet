def try_calculate_all_parameters(pred, target, smooth=1e-10, threshold=0.5):
    pred = pred.contiguous()
    target = target.contiguous()

    pred = (pred > threshold)
    target = (target > threshold)

    TP = (pred & target).sum().float()
    FP = (pred & ~target).sum().float()
    TN = (~pred & ~target).sum().float()
    FN = (~pred & target).sum().float()

    # 计算指标
    IOU = (TP + smooth) / (TP + FP + FN + smooth)
    DICE_Coefficient = (2 * TP + smooth) / (2 * TP + FP + FN + smooth)
    Precision = (TP + smooth) / (TP + FP + smooth)
    Recall = (TP + smooth) / (TP + FN + smooth)
    F1_score = (2 * Precision * Recall + smooth) / (Precision + Recall + smooth)
    foreground_accuracy = (TP + smooth) / (TP + FN + smooth)
    background_accuracy = (TN + smooth) / (TN + FP + smooth)
    Accuracy = (foreground_accuracy + background_accuracy) / 2
    # 返回计算得到的所有指标
    return Accuracy, Precision, Recall, F1_score, DICE_Coefficient,IOU, TP, FP, TN, FN


def evaluate_metrics(pred, target):
    # 确保预测和目标张量至少是二维的
    assert pred.dim() >= 2, "Prediction tensor must be at least 2D."
    assert target.dim() >= 2, "Target tensor must be at least 2D."

    # 移除不必要的维度，确保张量是二维的
    while pred.dim() > 2:
        pred = pred.squeeze(0)  # 假设额外的维度是在最前面
    while target.dim() > 2:
        target = target.squeeze(0)  # 同上

    # 现在，确保张量是二维的，然后获取其宽度和高度
    assert pred.dim() == 2, "Pred tensor must be 2D after squeezing."
    assert target.dim() == 2, "Target tensor must be 2D after squeezing."
    # height, width = pred.shape
    # height1, width1 = target.shape
    # print("pred",height, width,"target",height1, width1)



    # 计算指标

    accuracy, precision, recall, f1, dice, iou, TP, FP, TN, FN = try_calculate_all_parameters(pred, target)
    #↓↓↓↓用于检测混淆矩阵↓↓↓↓
    # print("Confusion Matrix:")
    # print(f"TP: {TP}, FP: {FP}")
    # print(f"FN: {FN}, TN: {TN}")
    return accuracy, precision, recall, f1, dice, iou, TP, FP, TN, FN   # acc, prec, rec, f1, dice, iou
