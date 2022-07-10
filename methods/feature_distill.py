import torch.nn.functional as F
import math
import torch

def att_val_kl(student_atts, student_vals, teacher_atts, teacher_vals, layer_selection):
    #TODO: 把这个fp16 32， 正规化，以及看amp方案
    loss_att = 0.
    loss_value = 0.

    batch_size, num_head, length, dk = student_vals[0].shape
    dk_sqrt = math.sqrt(dk)

    layer_selection = [int(item) for item in layer_selection.split(',')]

    new_teacher_atts = [teacher_atts[i] for i in layer_selection]

    #TODO: change to softmax and log 
    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (batch_size * num_head * length) #, reduction='batchmean', log_target=True)
        # loss_kl_tmp = F.mse_loss(student_att, teacher_att)
        loss_att += loss_kl_tmp

    new_teacher_value = [teacher_vals[i] for i in layer_selection]

    for student_value, teacher_value in zip(student_vals, new_teacher_value):
        vr_student = F.log_softmax(torch.bmm(student_value.reshape(-1, length, dk), student_value.reshape(-1, length, dk).transpose(1,2))/dk_sqrt, dim=-1)
        vr_teacher = F.softmax(torch.bmm(teacher_value.reshape(-1, length, dk), teacher_value.reshape(-1, length, dk).transpose(1,2))/dk_sqrt, dim=-1)

        loss_value_tmp = F.kl_div(vr_student, vr_teacher, reduction='sum')/(batch_size * num_head * length)
        # loss_value_tmp = F.mse_loss(vr_student, vr_teacher)
        loss_value += loss_value_tmp
    # loss  = loss_att + loss_value
    return loss_att, loss_value