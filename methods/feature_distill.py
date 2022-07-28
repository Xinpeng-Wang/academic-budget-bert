import torch.nn.functional as F
import math
import torch
from pretraining.utils import master_process
# import wandb
from .pear_loss import Dist_att

dist_att = Dist_att()

def att_val_kl(student_atts, student_qkv, teacher_atts, teacher_qkv, layer_selection):
    #TODO: 把这个fp16 32， 正规化，以及看amp方案
    loss_att = 0.
    loss_value = 0.

    batch_size, num_head, length, dk = student_qkv[0][2].shape
    dk_sqrt = math.sqrt(dk)
    layer_selection = [int(item) for item in layer_selection.split(',')]

    new_teacher_atts = [teacher_atts[i] for i in layer_selection]
    if type(layer_selection) is not list: 
        student_atts = [student_atts[-1]]
    #TODO: change to softmax and log 
    for student_att, teacher_att in zip(student_atts, new_teacher_atts):
        # batch_size, head_num, lenght = student_att.shape[0], student_att.shape[1], student_att.shape[2]
        student_att = F.log_softmax(student_att, dim=-1)
        teacher_att = F.softmax(teacher_att, dim=-1)
        loss_kl_tmp = F.kl_div(student_att, teacher_att, reduction='sum')/ (batch_size * num_head * length) #, reduction='batchmean', log_target=True)
        # loss_kl_tmp = F.mse_loss(student_att, teacher_att)
        loss_att += loss_kl_tmp

    new_teacher_value = [teacher_qkv[i][2] for i in layer_selection]
    if type(layer_selection) is not list:
        student_vals = [student_qkv[-1][2]]
    else:
        student_vals = [qkv[2] for qkv in student_qkv]
    for student_value, teacher_value in zip(student_vals, new_teacher_value):
        vr_student = F.log_softmax(torch.bmm(student_value.reshape(-1, length, dk), student_value.reshape(-1, length, dk).transpose(1,2))/dk_sqrt, dim=-1)
        vr_teacher = F.softmax(torch.bmm(teacher_value.reshape(-1, length, dk), teacher_value.reshape(-1, length, dk).transpose(1,2))/dk_sqrt, dim=-1)

        loss_value_tmp = F.kl_div(vr_student, vr_teacher, reduction='sum')/(batch_size * num_head * length)
        # loss_value_tmp = F.mse_loss(vr_student, vr_teacher)
        loss_value += loss_value_tmp
    # loss  = loss_att + loss_value
    return loss_att, loss_value




def minilm_v2(student_atts, student_qkv, teacher_atts, teacher_qkv, layer_selection):
    loss_q = 0.
    loss_k = 0. 
    loss_v = 0.

    bs_t, n_h_t, l_t, dk_t = teacher_qkv[0][2].shape
    dk_sqrt_t = math.sqrt(dk_t)

    layer_selection = [int(item) for item in layer_selection.split(',')]
    new_teacher_q = [teacher_qkv[i][0] for i in layer_selection]
    new_student_q = [student_qkv[-1][0].reshape([bs_t, n_h_t, l_t, -1])]
    dk_s = new_student_q[0].shape[3]
    dk_sqrt_s = math.sqrt(dk_s)
    
    for student_q, teacher_q in zip(new_student_q, new_teacher_q):
        qr_student = F.log_softmax(torch.bmm(student_q.reshape(-1, l_t, dk_s), student_q.reshape(-1, l_t, dk_s).transpose(1,2))/dk_sqrt_s, dim=-1)
        qr_teacher = F.softmax(torch.bmm(teacher_q.reshape(-1, l_t, dk_t), teacher_q.reshape(-1, l_t, dk_t).transpose(1,2))/dk_sqrt_t, dim=-1)
        loss_q_tmp = F.kl_div(qr_student, qr_teacher, reduction='sum')/(bs_t * n_h_t * l_t)
        loss_q += loss_q_tmp
    
    new_teacher_k = [teacher_qkv[i][1].transpose(2, 3) for i in layer_selection]
    new_student_k = [student_qkv[-1][1].transpose(2, 3).reshape([bs_t, n_h_t, l_t, -1])]

    for student_k, teacher_k in zip(new_student_k, new_teacher_k):
        kr_student = F.log_softmax(torch.bmm(student_k.reshape(-1, l_t, dk_s), student_k.reshape(-1, l_t, dk_s).transpose(1,2))/dk_sqrt_s, dim=-1)
        kr_teacher = F.softmax(torch.bmm(teacher_k.reshape(-1, l_t, dk_t), teacher_k.reshape(-1, l_t, dk_t).transpose(1,2))/dk_sqrt_t, dim=-1)
        loss_k_tmp = F.kl_div(kr_student, kr_teacher, reduction='sum')/(bs_t * n_h_t * l_t)
        loss_k += loss_k_tmp
    
    new_teacher_v = [teacher_qkv[i][2] for i in layer_selection]
    new_student_v = [student_qkv[-1][2].reshape([bs_t, n_h_t, l_t, -1])]

    for student_v, teacher_v in zip(new_student_v, new_teacher_v):
        vr_student = F.log_softmax(torch.bmm(student_v.reshape(-1, l_t, dk_s), student_v.reshape(-1, l_t, dk_s).transpose(1,2))/dk_sqrt_s, dim=-1)
        vr_teacher = F.softmax(torch.bmm(teacher_v.reshape(-1, l_t, dk_t), teacher_v.reshape(-1, l_t, dk_t).transpose(1,2))/dk_sqrt_t, dim=-1)
        loss_v_tmp = F.kl_div(vr_student, vr_teacher, reduction='sum')/(bs_t * n_h_t * l_t)
        loss_v += loss_v_tmp

    return  loss_q, loss_k, loss_v

def att_val_frame(teacher, student, args, batch, global_step, wandb, eval=False):
    log = 'eval' if eval else 'train'

    with torch.no_grad():
        attentions_teacher, qkv_teacher, prediction_score_teacher = \
                teacher(batch, output_attentions=True, output_qkv=True, output_loss=False)
    mlm_loss, attentions_st, qkv_st, prediction_score_st = \
            student.forward(batch, output_attentions=True, output_qkv=True, output_loss=True)
    if args.method == 'att_val_og':
        loss_att, loss_val = \
            att_val_kl(attentions_st, qkv_st, attentions_teacher, qkv_teacher, args.layer_selection)
        total_loss = loss_att + loss_val
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_att": loss_att}, step=global_step)
            wandb.log({f"{log}/loss_val": loss_val}, step=global_step)
    elif args.method == 'minilm_v2':
        loss_q, loss_k, loss_v = \
            minilm_v2(attentions_st, qkv_st, attentions_teacher, qkv_teacher, args.layer_selection)
        total_loss = loss_q + loss_k + loss_v        
        
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_q": loss_q}, step=global_step)
            wandb.log({f"{log}/loss_k": loss_k}, step=global_step)
            wandb.log({f"{log}/loss_v": loss_v}, step=global_step)
    elif args.method == 'pear_col':
        inter_token_1, inter_token_2, inter_head, inter_sentence = dist_att.forward(attentions_teacher[-1], attentions_st[-1])
        total_loss = inter_token_1 + inter_token_2 + inter_head + inter_sentence
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/inter_token_1": inter_token_1}, step=global_step)
            wandb.log({f"{log}/inter_token_2": inter_token_2}, step=global_step)
            wandb.log({f"{log}/inter_head": inter_head}, step=global_step)
            wandb.log({f"{log}/inter_sentence": inter_sentence}, step=global_step)
    return total_loss


def twostage(teacher, student,args, batch, time_diff, global_step, wandb, eval=False):
    log = 'eval' if eval else 'train'
    time_proportion = time_diff / args.total_training_time
    if time_proportion < 0.5 :
        mlm_loss , attentions_st, values_st, prediction_score_st = \
            student.forward(batch, output_attentions=True, output_qkv=True, output_loss=True)
        if master_process(args):
            wandb.log({f"{log}/loss": mlm_loss}, step=global_step)
        total_loss = mlm_loss
    else:
        with torch.no_grad():
            attentions_teacher, values_teacher, prediction_score_teacher = \
                teacher(batch, output_attentions=True, output_qkv=True, output_loss=False)
        mlm_loss , attentions_st, values_st, prediction_score_st = \
            student.forward(batch, output_attentions=True, output_qkv=True, output_loss=True)

        loss_att, loss_val = \
                att_val_kl(attentions_st, values_st, attentions_teacher, values_teacher, args.layer_selection)
        total_loss = loss_att + loss_val
        if master_process(args):
            wandb.log({f"{log}/loss": total_loss}, step=global_step)
            wandb.log({f"{log}/loss_att": loss_att}, step=global_step)
            wandb.log({f"{log}/loss_val": loss_val}, step=global_step)
    return total_loss 


