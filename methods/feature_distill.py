import torch.nn.functional as F
import math
import torch
from pretraining.utils import master_process
# import wandb

def att_val_kl(student_atts, student_vals, teacher_atts, teacher_vals, layer_selection):
    #TODO: 把这个fp16 32， 正规化，以及看amp方案
    loss_att = 0.
    loss_value = 0.

    batch_size, num_head, length, dk = student_vals[0].shape
    dk_sqrt = math.sqrt(dk)

    layer_selection = [int(item) for item in layer_selection.split(',')]

    new_teacher_atts = [teacher_atts[i] for i in layer_selection]
    student_atts = [student_atts[-1]]
    #TODO: change to softmax and log 
    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (batch_size * num_head * length) #, reduction='batchmean', log_target=True)
        # loss_kl_tmp = F.mse_loss(student_att, teacher_att)
        loss_att += loss_kl_tmp

    new_teacher_value = [teacher_vals[i] for i in layer_selection]

    student_vals = [student_vals[-1]]
    for student_value, teacher_value in zip(student_vals, new_teacher_value):
        vr_student = F.log_softmax(torch.bmm(student_value.reshape(-1, length, dk), student_value.reshape(-1, length, dk).transpose(1,2))/dk_sqrt, dim=-1)
        vr_teacher = F.softmax(torch.bmm(teacher_value.reshape(-1, length, dk), teacher_value.reshape(-1, length, dk).transpose(1,2))/dk_sqrt, dim=-1)

        loss_value_tmp = F.kl_div(vr_student, vr_teacher, reduction='sum')/(batch_size * num_head * length)
        # loss_value_tmp = F.mse_loss(vr_student, vr_teacher)
        loss_value += loss_value_tmp
    # loss  = loss_att + loss_value
    return loss_att, loss_value


def att_val_frame(teacher, student, args, batch):
    with torch.no_grad():
        attentions_teacher, values_teacher, prediction_score_teacher = \
                teacher(batch, output_attentions=True, output_values=True, output_loss=False)
    mlm_loss , attentions_st, values_st, prediction_score_st = \
            student.forward(batch, output_attentions=True, output_values=True, output_loss=True)

    loss_att, loss_val = \
            att_val_kl(attentions_st, values_st, attentions_teacher, values_teacher, args.layer_selection)
    
    return loss_att, loss_val

def twostage(teacher, student,args, batch, time_diff, global_step, wandb):
    time_proportion = time_diff / args.total_training_time
    if time_proportion < 0.5 :
        mlm_loss , attentions_st, values_st, prediction_score_st = \
            student.forward(batch, output_attentions=True, output_values=True, output_loss=True)
        if master_process(args):
            wandb.log({"train/loss": mlm_loss}, step=global_step)
        total_loss = mlm_loss
    else:
        with torch.no_grad():
            attentions_teacher, values_teacher, prediction_score_teacher = \
                teacher(batch, output_attentions=True, output_values=True, output_loss=False)
        mlm_loss , attentions_st, values_st, prediction_score_st = \
            student.forward(batch, output_attentions=True, output_values=True, output_loss=True)

        loss_att, loss_val = \
                att_val_kl(attentions_st, values_st, attentions_teacher, values_teacher, args.layer_selection)
        total_loss = loss_att + loss_val
        if master_process(args):
            wandb.log({"train/loss": total_loss}, step=global_step)
            wandb.log({"train/loss_att": loss_att}, step=global_step)
            wandb.log({"train/loss_val": loss_val}, step=global_step)
    return total_loss 