import os

class evaluator():
    def __init__(self, label_list):
        #self.confusion_mat = [[0]* num_of_class]]*num_of_class
        num_of_class = len(label_list)
        self.label_list = label_list
        self.num_of_class = len(label_list)
        self.true_label_count = [0]* self.num_of_class
        self.pred_label_count = [0]* self.num_of_class
        self.correct_label_count = [0]* self.num_of_class

        self.class_precision = [0]* self.num_of_class
        self.class_recall = [0]* self.num_of_class
        self.class_f1 =[0]* self.num_of_class

        self.recall = 0
        self.precision = 0
        self.f1 = 0

        self.recall_2 = 0
        self.precision_2 = 0
        self.f1_2 = 0

    def reset(self):

        self.true_label_count = [0]* self.num_of_class
        self.pred_label_count = [0]* self.num_of_class
        self.correct_label_count = [0]*self.num_of_class

        self.class_precision = [0]* self.num_of_class
        self.class_recall = [0]* self.num_of_class
        self.class_f1 =[0]* self.num_of_class

        self.recall = 0
        self.precision = 0
        self.f1 = 0


    def per_image_cal_topk(self, predicted, target):
        Relevant_Items_Recommended =0
        for p in predicted:
            if p in target:
                Relevant_Items_Recommended  = Relevant_Items_Recommended + 1

        recall = Relevant_Items_Recommended/len(target)
        pre = Relevant_Items_Recommended/len(predicted)
        self.recall_2 = self.recall_2 + recall
        self.precision_2 = self.precision_2 + pre



    def per_image_cal(self, predicted, target):
        #print("P======== {}".format(predicted))
        #print("G======== {}".format(target))
        for p in predicted:
            self.pred_label_count[p] = self.pred_label_count[p]+1
            if p in target:
            #if target[p] > 0:
                #print("{} is in G======== {}".format(p,target))
                #print("P====={}".format(predicted))
                self.correct_label_count[p] = self.correct_label_count[p] +1

        for t in target:
            #if target[t] > 0:
            self.true_label_count[t] = self.true_label_count[t] +1


    def f_1_score(self, recall, precision):
        if recall + precision !=0:
            f_1 = 2*(recall* precision) / (recall + precision)
            return f_1
        return 0

    def print_result(self, cr = False, cp = False, cf=False, r= True, pr=True, f=True, dump = None):
        full_string = 'Result \n-------------------------\n'
        '''
        label_string = 'labels'.ljust(10)+"\t\t"
        for l in self.label_list:
            label_string = label_string + l.ljust(10)

        full_string = full_string + label_string + '\n'

        if cp == True:
            cp_string = 'class_prec'.ljust(10)+"\t\t"
            for p in self.class_precision:
                cp_string = cp_string + str("{:.2f}".format(p)).ljust(10)
            full_string = full_string + cp_string + '\n'

        if cr == True:
            cr_string = 'class_rc'.ljust(10)+"\t\t"
            for p in self.class_recall:
                cr_string = cr_string + str("{:.2f}".format(p)).ljust(10)
            full_string = full_string + cr_string + '\n'
        '''

        if cf == True:
            cf_string = 'class_f1'.ljust(10)+"\t\t"
            for p in self.class_f1:
                cf_string = cf_string + str("{:.2f}".format(p)).ljust(10)
            full_string = full_string + cf_string + '\n'

        if r == True:
            full_string = full_string + 'recall:'.ljust(10) + str(self.recall).ljust(10) + '\n'
        if pr == True:
            full_string = full_string + 'precision:'.ljust(10) + str(self.precision).ljust(10) + '\n'
        if f == True:
            full_string = full_string + 'f_1 score:'.ljust(10) + str(self.f1).ljust(10) + '\n'


        if r == True:
            full_string = full_string + 'recall:'.ljust(10) + str(self.recall_2).ljust(10) + '\n'
        if pr == True:
            full_string = full_string + 'precision:'.ljust(10) + str(self.precision_2).ljust(10) + '\n'


        if dump != None:
            file1 = open(dump,'w')
            file1.write(full_string)
            file1.close()
            print('result is being saved as ' + dump)

        print(full_string)


    def evaluate(self, predict, target, dump = None):

        self.reset()

        for pred, tar in zip(predict,target):
            self.per_image_cal(pred,tar)
            self.per_image_cal_topk(pred,tar)

        self.class_recall = [cl/tl if tl!=0 else 0 for cl,tl in zip(self.correct_label_count,self.true_label_count )]
        self.class_precision = [cl/pl if pl!=0 else 0 for cl, pl in zip(self.correct_label_count, self.pred_label_count )]
        self.class_f1 = [ self.f_1_score(r,p) for r,p in zip(self.class_recall,self.class_precision )]

        self.recall = sum(self.correct_label_count)/ sum(self.true_label_count )
        self.precision = sum(self.correct_label_count)/ sum(self.pred_label_count)
        self.f1 = self.f_1_score(self.recall , self.precision )

        self.recall_2 = self.recall_2/len(predict)
        self.precision_2 = self.precision_2/len(predict)

        self.print_result(dump = dump)
