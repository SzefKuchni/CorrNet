#read this into R
text_LR = readLines('C:/Users/T540pDLEWYNBQ/Google Drive/Wszystko/Inne/Studia doktoranckie - Informatyka/CorrNet/CorrNet/test_LR.txt')

#serach for view1 to view2
idx<-grep('view1 to view2', text_LR)
view1_to_view2<-as.numeric(text_LR[idx+1])

#serach for view2 to view1
idx<-grep('view2 to view1', text_LR)
view2_to_view1<-as.numeric(text_LR[idx+1])

#serach for validation correlation
idx<-grep('validation correlation', text_LR)
validation_correlation<-as.numeric(text_LR[idx+1])

#serach for test correlation
idx<-grep('test correlation', text_LR)
test_correlation<-as.numeric(text_LR[idx+1])

result_LR<-data.frame("Type"="LR",
                   "view1_to_view2"=view1_to_view2,
                   "view2_to_view1"=view2_to_view1,
                   "validation_correlation"=validation_correlation,
                   "test_correlation"=test_correlation)

#read this into R
text_UD = readLines('C:/Users/T540pDLEWYNBQ/Google Drive/Wszystko/Inne/Studia doktoranckie - Informatyka/CorrNet/CorrNet/test_UD.txt')

#serach for view1 to view2
idx<-grep('view1 to view2', text_UD)
view1_to_view2<-as.numeric(text_UD[idx+1])

#serach for view2 to view1
idx<-grep('view2 to view1', text_UD)
view2_to_view1<-as.numeric(text_UD[idx+1])

#serach for validation correlation
idx<-grep('validation correlation', text_UD)
validation_correlation<-as.numeric(text_UD[idx+1])

#serach for test correlation
idx<-grep('test correlation', text_UD)
test_correlation<-as.numeric(text_UD[idx+1])

result_UD<-data.frame("Type"="UD",
                   "view1_to_view2"=view1_to_view2,
                   "view2_to_view1"=view2_to_view1,
                   "validation_correlation"=validation_correlation,
                   "test_correlation"=test_correlation)

result<-rbind(result_LR,result_UD)
write.csv(result, "experiment_results.csv", row.names = F)
