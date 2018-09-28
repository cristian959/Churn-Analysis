#to manipulate data
library("tidyverse")
library("janitor")
library("caret")
library("rpart")
library("rpart.plot")
library("randomForest")
library("ROSE")
library("e1071")
library("plotly")
library("IRdisplay")
library("ggplot2")
library("plyr")

#We import data and look at quick insight results. We encode the SeniorCitizen variable as a factor variable.
db_churn = read_csv(file.choose())

#To see variables structure 
str(db_churn)

summary(db_churn)


prop = tabyl(db_churn$Class)
prop

sum(is.na(db_churn))

sapply(db_churn, function(x) sum(is.na(x)))

db_churn <- db_churn[complete.cases(db_churn), ]

sum(is.na(db_churn))


#Create the partition (data train / data test)
set.seed(7)
trainId = createDataPartition(db_churn$Class, 
                              p=0.7, list=FALSE,times=1)

db_train = db_churn[trainId,]
db_test = db_churn[-trainId,]


####################################################################################
#We have to normalize the numerical variables

normalize = function(x) {
  result = (x - min(x, na.rm = TRUE)
  ) / (max(x, na.rm = TRUE) - min(x, na.rm = TRUE))
  return(result)
}
norm.train = lapply(db_train %>% 
                      select(network_age, Aggregate_Total_Rev, Aggregate_SMS_Rev,Aggregate_Data_Rev,
                             Aggregate_Data_Vol,Aggregate_Calls,Aggregate_ONNET_REV,Aggregate_OFFNET_REV,
                             Aggregate_complaint_count),
                    normalize)
norm.train = do.call(cbind, norm.train) %>%
  as.data.frame()


##############################################################################################
#Decision tree

tree = rpart(Class ~., data = db_train, method="class")
rpart.plot(tree)


#Random forest
ctrl = trainControl(method = "cv", number=5, 
                    classProbs = TRUE, summaryFunction = twoClassSummary)

model.rf = train(Class ~., data = db_train,
                 method = "rf",
                 ntree = 75,
                 tuneLength = 5,
                 metric = "ROC",
                 trControl = ctrl)

model.rf

db_train$ChurnNum = ifelse(db_train$Class == "Churned",1,0)
good_model = step(glm(ChurnNum ~.,data = db_train %>% 
                        select(-Class ), 
                      family=binomial(link='logit')), 
                  direction="both")
#Details about the model
summary(good_model)

norm.test = lapply(db_test %>% select(network_age,Aggregate_Total_Rev, Aggregate_SMS_Rev,
                                      Aggregate_Data_Rev,Aggregate_Data_Vol,Aggregate_Calls,
                                      Aggregate_ONNET_REV,Aggregate_OFFNET_REV,Aggregate_complaint_count), normalize)
norm.test = do.call(cbind, norm.test) %>%
  as.data.frame()

#Convert churn to numerical
db_test$ChurnNum  = ifelse(db_test$Class == "Churned", 1, 0)

#Make predictions
pred_tree = predict(tree, db_test, type = c("prob"))[,2]
pred_rf = predict(object=model.rf, db_test, type='prob')[,2]
pred_logistic = predict(good_model, db_test, type="response")

#Plot the ROC curve abaout our models
roc_tree = roc.curve(response = db_test$ChurnNum, pred_tree, 
                     col = "#0d84da")
roc_rf = roc.curve(response = db_test$ChurnNum, pred_rf, 
                   col = "#ef0a30", add.roc=TRUE)
roc_logistic = roc.curve(response = db_test$ChurnNum, pred_logistic, 
                         col = "#45a163", add.roc=TRUE)

legend("bottomright", legend=c("Decision Tree", "Random Forest", 
                               "Logistic Regression"), 
       lwd = 2, col = c("#0d84da", "#ef0a30", "#45a163"))

#Each customer has a score that corresponds to his probability to churn. 
#Let's see the score of the five first customers of our database.
head(pred_logistic,5)

#To optimize the threshold, we want to compute different statistical indicators 
#(accuracy, precision, sensitivity, f1 and kappa) for different threshold values.
comp = cbind.data.frame(answer = db_test$ChurnNum, 
                        pred=pred_logistic) %>%
  arrange(desc(pred))

indic_perf = function(x){
  compare = comp %>%
    mutate(pred = ifelse(pred>x,1,0))
  
  if(ncol(table(compare))>1){
    
    mat = confusionMatrix(table(compare), positive = "1")
    #acuracy 
    acc = mat$overall["Accuracy"]
    
    #Kappa
    kap = mat$overall["Kappa"]
    
    #sensitivity
    sen = mat$byClass["Sensitivity"]
    
    #F1
    f1_stat = mat$byClass["F1"]
    
    #Precision
    prec = mat$byClass["Precision"]
    
    
  }else{
    acc = NA
    prec = NA
    sen = NA 
    kap = NA
    f1_stat = NA
  }
  return(data.frame(threshold = x, accuracy = acc, 
                    precision = prec, sensitivity = sen, 
                    kappa = kap, f1= f1_stat))
}
indics = do.call(rbind, lapply(seq(0.05,0.95, by=0.001), 
                               indic_perf)) %>%
  filter(!is.na(accuracy))



gather_indics = tidyr::gather(indics, variable, 
                              value, -threshold) %>%
  group_by(variable) %>%
  mutate(color =  (max(value) == value), 
         threshold = as.numeric(threshold) )

q=ggplot(gather_indics , aes(x= threshold, y=value)) +
  ggtitle("Indicator values by thresholds")+
  geom_point(aes(color = color), size=0.5) +
  facet_wrap(~variable, scales = 'free_x') +
  scale_color_manual(values = c(NA, "tomato")) +
  labs(x="thresholds", y=" ") +
  geom_line(color="navy") + theme_bw()+ 
  theme( legend.position="none")
offline(ggplotly(q),  width = '100%')

#We draw a table where we show the maximum value for each indicator.
max_indics = indics %>%
  filter(accuracy == max(accuracy, na.rm=TRUE) | precision == max(precision, na.rm = TRUE) | sensitivity == max(sensitivity, na.rm = TRUE) | kappa == max(kappa, na.rm = TRUE) | f1 == max(f1, na.rm = TRUE) )

max_indics

#We still have a lot of values so we decide to remove some rows. We keep the five most relevant thresholds.
max_indics %>%
  filter( threshold %in% c("0.050", "0.051", "0.436", "0.437", "0.548"))


#At this stage, we can choose a threshold based on what we think is the most important. I suggest to continue
#with 0.548. Hence we have the following confusion matrix.
compare = comp %>%
  mutate(pred = ifelse(pred>0.548,1,0))
confusionMatrix(table(compare), positive = "1")

fivtile = nrow(comp)/20
step = floor(fivtile * 1:20)
pct = sapply(step, function(x){
  return(mean(comp[1:x,1]))})

lift = data.frame(label= seq(from = 5, to = 100, by=5), score = pct*100)
q = ggplot(lift, aes(x=label, y=score))+
  
  geom_bar(stat="identity",position="stack",color="navy", fill="navy")+ 
  ggtitle("Churn rate per cumulative percentile of \n customers  with the highest probability to leave")+
  coord_cartesian(xlim = c(5,100), ylim = c(0,100))+
  scale_y_continuous(breaks = seq(from = 0, to = 100, by=25), labels = function(x) paste0(x,"%", sep = ""))+
  scale_x_continuous(breaks = c(5, 25, 50, 75, 100), labels = function(x) paste0(x,"%", sep = ""))+
  labs(x="cumulative percentile ", y="churn rate") + 
  geom_line(aes(y=score[20]),linetype=2, size=1, color="tomato")+ 
  theme_minimal()+
  theme( 
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    plot.title = element_text(size=17,face="bold", hjust=0.5),
    axis.text.x = element_text(vjust=1),
    axis.ticks.x = element_blank(),
    axis.title.x = element_text(size=13, face="bold", vjust=-0.4),
    axis.title.y = element_text(size=13,face="bold")#,
    #strip.text.x = element_text(face="italic", size=11)
  )

print(q)

# Conclusions
# 1.- Aggregate_Total_Rev is the best metric to show fidelity, and it has an inverse relation with Churn. 
#  That suggest that main customers are satisfied
# 2.- Aggregate_SMS_Rev has a direct relation with Churn, that suggests that these customers who use an old technology like SMS 
#  are not satisfied with the price they are paying.
# 3.- The same situation is the same with Aggregate_Data_Rev which has a direct relation with Churn. 
# Perhaps an expensive price for they are receiving.
# 4.- There is no colinearity into model because none correlations between main variable are over 75%
# 5.- network_age has a inverse relation that suggest that customers with more antiquity use stay at the company


var_rf <- varImp(model.rf, scale = FALSE)
var_rf
plot(var_rf, top = 20)

explained_model = step(glm(ChurnNum ~.,data = db_train %>% 
                             select(-Class, -starts_with("sep"),-starts_with("aug")),
                           family=binomial(link='logit')), 
                       direction="both")

summary(explained_model)

#We plot the correlation matrix of the variables
correlations <- db_train %>% 
  select(-Class, -starts_with("sep"),-starts_with("aug")) %>% cor()

highcorr <- findCorrelation(correlations, cutoff = .75)
correlations[,highcorr]

pred_explained = predict(object=explained_model, db_test %>% select(-Class, -starts_with("sep"),-starts_with("aug")),
                         type='response')

new_test = db_test %>% select(-Class, -starts_with("sep"),-starts_with("aug"))

roc_explained = roc.curve(response = new_test$ChurnNum, pred_explained, 
                          col = "#ef0a30", add.roc=TRUE)
roc_explained

