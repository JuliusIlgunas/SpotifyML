library(e1071)
library(party)
library(caret)
library(nnet)
library(dplyr)
library(randomForest)
library(glmnet)
library(ROCR)
library(ggplot2)

#Reading data
df <- read.csv("data.csv")
str(df)

#Selected features
df1 <- df %>% select(c("target", "acousticness", "danceability", "energy", "instrumentalness", "liveness",
                       "loudness", "speechiness", "tempo", "valence"))

#Renaming the target values
df1$target[df1$target == 1] <- "Liked"
df1$target[df1$target == 0] <- "Disliked"

#Chagning the loudness values, from negative to positive. For better interpretation
DF <- transform(df1, loudness = as.numeric(abs(loudness)), target = as.factor(target))

#Removing missing values
data <- DF[complete.cases(DF),] 
dim(data)
str(data)


#Gathering all of distinct artists, number of their songs and looking at what is the average value of target
aa <-  df %>% count(artist)
artists <- df %>%  group_by(artist) %>% summarise_at(vars(target), funs(mean))

#selecting top 20 atrists with most songs
artists2 <- merge(aa, artists, by = "artist")
artists2 <- artists2[order(artists2$n, decreasing=TRUE), ]
artists2 <- artists2[1:20,]


#Graph of 20 artists with most songs and the "likeness" of each artist
g <- artists2 %>% ggplot(aes(artist, n, fill=target)) + geom_bar(stat = "identity") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
  scale_fill_gradient(low = "firebrick3", high = "forestgreen")


#Splitting the data into train(80%) and test(20%) datasets
set.seed(124)
train_size <- floor(0.8 * nrow(data))
train_ind <- sample(1:nrow(data), size = train_size)
train <- data[train_ind, ]
test <- data[-train_ind, ]

#Scatter plots of all the features, with different color for each target
pairs(log(data[,-1] + 1), col=c("red", "blue")[as.numeric(data[,1])])
data[,-1] <- log(data[,-1] + 1)


#SVM METHOD

#Tuning the cost and gamma parameters
tunefit <- tune.svm(target~., data=train, cost=10^(2:3), gamma= 10^(-3:-1))
summary(tunefit)
svmfit <- tunefit$best.model
yhat <- predict(svmfit,test)


#Fitting the SVM method on train data and making predictions for test data
svmfit1 <- svm(target ~ ., data=train, kernel="linear", cost=1000, gamma = tunefit$best.model$gamma,
               scale=FALSE, probability = TRUE)
summary(svmfit1)
pred1 <- predict(svmfit1, test)
pred1_r <- predict(svmfit1, test, probability = TRUE, type = "prob")
pred1_r <- data.frame(attributes(pred1_r))
pred1_r <- pred1_r$probabilities.Liked
#table(pred1, test$target)

#Confusion matrix and statistics after using SVM method 
confusionMatrix(table(pred1, test$target), positive = "Liked")

#Gathering the prediction scores
svm_scores <- cbind(rep("SVM",length(pred1_r)), pred1_r, test$target)

#DECISION TREE METHOD

#Fitting the decision tree model on train data and making predictions on test data
dt <- ctree(target ~ ., data=train, controls=ctree_control(maxdepth=4))
pred2 <- predict(dt, test)
pred2_r <- predict(dt, test, type = "prob")
pred2_r <- unlist(pred2_r)
pred2_r <- pred2_r[seq(2, length(pred2_r), 2)]
#mean(p == test$target)
#table(p, test$target)

#Confusion matrix and statistics after using decision tree method 
confusionMatrix(table(pred2, test$target), positive = "Liked")

#Gathering the prediction scores
dt_scores <- cbind(rep("DT",length(pred2_r)), pred2_r, test$target)

#Plotting the decision tree
#print(dt)
plot(dt, type="simple")
plot(dt, type="extended")


#RANDOM FOREST METHOD

#Tuning the mtry and ntree parameters
tune_rf <- tune.randomForest(target ~ ., data = train, mtry = 0:5, nodesize = TRUE, ntree = 100:200)
tune_rf

#Fitting the random forest model on train data and making predictions on test data
ran2 <- randomForest(target ~ . , data = train, ntree = tune_rf$best.model$ntree,
                     mtry = tune_rf$best.model$mtry, importance = TRUE)
plot(ran2)
pred3 <- predict(ran2, test)
pred3_r <- predict(ran2, test, type = "prob")
pred3_r <- as.data.frame(pred3_r)

#Confusion matrix and statistics after using random forest method 
confusionMatrix(table(pred3, test$target), positive = "Liked")

#Gathering the prediction scores
rf_scores <- cbind(rep("RF",length(pred3_r)), pred3_r$Liked, test$target)

#Checking the variable importance
importance(ran2)


#NEURAL NETWORK METHOD

#Tuning the size and decay parameters
tune_nn <- tune.nnet(target~., data=train, size=1:5, maxit=1000, decay = 0:5)
tune_nn

#Fitting the neural network model on train data and making predictions on test data
fit <- nnet(target~., train, size=tune_nn$best.model$n[2], maxit=1000, decay = tune_nn$best.model$decay)
pred4 <- predict(fit, test, type = "class") 
pred4_r <- predict(fit, test, type = "raw") 
#table(predict(fit, test, type="class"), test$target)
#table(pred, test$target)

#Confusion matrix and statistics after using neural network method 
confusionMatrix(table(pred4, test$target), positive = "Liked")

#Gathering the prediction scores
NN_scores <- cbind(rep("NN",length(pred4_r)), pred4_r, test$target)


#LOGISTIC REGRESSION

set.seed(123)
cv.lasso <- cv.glmnet(as.matrix(train[,-1]), train[,1], alpha = 1, family = "binomial")
plot(cv.lasso)

model_lasso <- glmnet(as.matrix(train[,-1]), train[,1], alpha = 1, family = "binomial",
                      lambda = cv.lasso$lambda.min)

#WITH LASSO REGULARIZATION
model_logit <- glm(target~., data=train, family = "binomial")

summary(model_logit)

#Making predictions for test data with logit model
pred <- predict(model_logit, test, type = "response")
pred.logit <- rep('Disliked',length(pred))
pred.logit[pred>=0.5] <- 'Liked'

#Making predictions for test data with lasso model
probabilities <- model_lasso %>% predict(newx = as.matrix(test[-1]))
pred.lasso <- rep('Disliked',length(probabilities))
pred.lasso[probabilities>=0.5] <- 'Liked'

#Confusion matrix and statistics after using logistic regression
confusionMatrix(table(pred.logit, test$target), positive = "Liked")

#Confusion matrix and statistics after using logistic regression with lasso regularization 
confusionMatrix(table(pred.lasso, test$target), positive = "Liked")

#Gathering the prediction scores
log_scores <- cbind(rep("Log",length(pred)), pred, test$target)
lasso_scores <- cbind(rep("Lasso",length(probabilities)), probabilities, test$target)


#Putting the prediction scores together in one data frame
models <- data.frame(rbind(svm_scores, dt_scores, rf_scores, NN_scores, log_scores, lasso_scores))
myModels <- levels(models$V1)
colnames(models) <- c("Model", "Score", "Target")
models$Target <- ifelse(models$Target == 2, TRUE, FALSE)
models$Model <- as.character(models$Model)


# Plotting the ROC curves and calculating AUC

plot(1, type="n", xlab="False Positive Rate", ylab="True Positive Rate",
     xlim=c(0, 1), ylim=c(0, 1), main="ROC")
au <- NULL
mod_cols <- c("red", "blue", "green", "purple", "orange", "cyan")
for (i in 1:length(myModels)) {
  
  mod <- filter(models, Model == myModels[i])
  pred <- prediction(mod$Score, mod$Target)
  roc <- performance(pred, "tpr", "fpr")
  plot(roc, add=T, col=mod_cols[i])
  
  auc <- performance(pred, "auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc, 4)
  aucc <- c(au, auc)
  au <- aucc
}
auccc <- paste(myModels, aucc, sep=" ")
legend(.79, .42, legend = auccc, title = "AUC", col=c("red", "blue", "green", "purple", "orange", "cyan"), lty=1, cex=0.8)
abline(a=0, b=1)
models$Score <- as.numeric(as.character(models$Score))

