import torch.nn as nn
import torch
import torch.nn.functional as F


class Mixture_loss(nn.Module):
    def __init__(self, num_of_mixture = 16):
        super(Mixture_loss, self).__init__()
        self.label_loss = nn.BCELoss()
        self.num_of_mixture = num_of_mixture

    def forward(self,g_x,p_yi_x, labels):

        #p_yi_x = p_yi_x.view(p_yi_x.size(0),-1,self.num_classes) # bs x K x num_of_class
            #print('p_yi_x shape {}'.format(p_yi_x.size()))
        #with torch.no_grad():
        label = labels.detach().unsqueeze(dim =1) #bs x 1 x num clas
        label = torch.repeat_interleave(label, repeats=self.num_of_mixture, dim=1) #bs x self.num_of_mixture x num clas
        label_c = 1 - label

        p_yi_x_c = 1 - p_yi_x
        #print('label shape {}'.format(label.size()))
        #print('label  {}'.format(label ))
        p_yi_x1 = p_yi_x * label + label_c * p_yi_x_c
        #print('pyix after {}'.format(p_yi_x))
        p_y_x = torch.prod(p_yi_x1, dim = -1 )#bs x num of mix
        #print('pyx shape {}'.format(p_y_x.size()))
        #print(p_y_x.type())
        h_x = g_x * p_y_x
        #print(h_x)
        h_x = F.softmax(h_x, dim = -1) #bs x num of mix

            #print('h_x {}'.format(h_x.size()))
            #print(h_x)
            #return g_x, p_y_x, h_x, p_yi_x

        p_y = p_y_x .unsqueeze(dim =-1)
        log_py = torch.log(p_y) #n x k x 1 if not unsqueeze
        #print('log py {}'.format(p_y))

        g_x = g_x.unsqueeze(dim =-1)
        log_gx = torch.log(g_x) #n x k x 1
        #print('log gx {}'.format(log_gx))

        h_x = h_x.unsqueeze(dim = -1)
        hx_t = torch.transpose(h_x, 1, 2).contiguous()
        #print(h_x)
        first_sum = torch.bmm(hx_t,log_gx) + torch.bmm(hx_t, log_py) #n x 1 x 1 # may need to squeeze

        #print('from loss fs {}'.format(first_sum.size()))
        first_sum = first_sum.squeeze(dim=-1).squeeze(dim=-1)

        #print('from loss fs {}'.format(first_sum.size()))
        all_sum = torch.sum(first_sum, dim = 0)*-1 # -1 for NLL
        #print('from loss as {}'.format(all_sum))

        #shape of gx should be n x k x 1
        gx = g_x.detach()
        class_prob = p_yi_x * gx#torch.bernoulli(p_yi_x) * gx
        #print('from loss class prob1 {}'.format(class_prob.size()))
        class_prob = torch.sum(class_prob,dim=1)
        #print('from loss class prob2 {}'.format(class_prob.size()))
        #print('from loss class lbls {}'.format(labels.size()))
        class_loss = self.label_loss(class_prob, labels.double())
        #print('class loss {}'.format(class_loss))
        return all_sum + class_loss, all_sum.item(), class_loss.item()
