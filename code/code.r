# http://www.ats.ucla.edu/stat/r/dae/logit.htm
# http://stats.stackexchange.com/questions/25389/obtaining-predicted-values-y-1-or-0-from-a-logistic-regression-model-fit

# source("c:/alexey_workspace/sauder/kaggle/data-science-titanic-learn/code/code.R")
library(car)

fnFindOptimalThreshold = function(g1Z,g1ZPredicted, nSteps = 1000){

	rOptimalThreshold	= 0.5
	rOptimalDist 		= 1.0
	
	arThreshold	= seq(0.0,1.0,length = nSteps)

	for(i in 1:length(arThreshold))
	{	
		rThreshold = arThreshold[i]
		
		idx		= which(g1Z == 1)
		g1ZHat	= (g1ZPredicted > rThreshold)

		rClassificationRate	= mean(g1Z == g1ZHat, na.rm = TRUE)
		rSensitivity		= mean(g1ZHat[idx] == 1, na.rm = TRUE)
		rSpecificity		= mean(g1ZHat[-idx] == 0, na.rm = TRUE)
		
		rDistToMinimize = cbind(rSensitivity,rSpecificity)-c(1,1)
		rDistToMinimize = sqrt( rDistToMinimize[1]^2 + rDistToMinimize[2]^2 ) 
		
		if(rDistToMinimize <= rOptimalDist)
		{
		  rOptimalDist 		= rDistToMinimize;
		  rOptimalThreshold	= rThreshold;
		}
   }

	return (rOptimalThreshold)
}

fnCabinTitleToCategory = function(x) {
  
  return (substr(x, 1, 1))
}
 
fnReshuffleData = function(data.set, strLog = ""){

	data.set$Pclass = as.factor(data.set$Pclass)
	data.set$Sex = as.factor(data.set$Sex)
	# data.set$Age = fnScaleData(data.set$Age, bExcludeNA)
	data.set$SibSp = as.factor(data.set$SibSp)
	data.set$Parch = as.factor(data.set$Parch)

	# data.set$Cabin = as.factor(sapply(data.set$Cabin, FUN = fnCabinTitleToCategory))
	data.set$Cabin = as.factor(fnCabinTitleToCategory(data.set$Cabin))
	data.set$Embarked = as.factor(data.set$Embarked)
	
	if(strLog != "")
		write.csv(data.set, file = strLog, quote = FALSE, row.names = FALSE)
	
	return (data.set)	
}

fnRemoveFactors = function(z,namesToRemove){

  return (z[ , -which(names(z) %in% namesToRemove)])
}

fnOutCorrPanel = function(x,y, digits = 2, prefix = ""){
  
	usr = par("usr"); on.exit(par(usr))
	par(usr = c(0,1,0,1))
	r = cor(x,y)
	txt = format(c(r,0.123456789),digits =digits)[1]
	txt = paste(prefix, txt, sep = "")
	text(.5,.5,txt)  
}

fnScalingUnset = function(z, mu, sigma){
  
  return (z*sigma + mu)
}

fnScaleData = function(z, bExcludeNA, mu = NA, sigma = NA){

	if(is.na(mu) | is.na(sigma))
	{
	  z.scaled = (z - mean(z, na.rm = bExcludeNA))/sd(z, na.rm = bExcludeNA)  
	}
	else
	{
	  z.scaled = (z - mu)/sigma  
	}    

	return (z.scaled)
}

fnSplitTrainingValidation = function(dataset, strOutput = "training"){

	n = dim(rawdata)[1]
	
	if(strOutput == "training")
	{
		dataset.training	= rawdata[1:floor(n*2/3),]
		return (dataset.training)		
	}
	else if (strOutput == "validation")
	{
		dataset.validation	= rawdata[1+floor(n*2/3):(n-1),]
		return (dataset.validation)		
	}
	else
		stop("ERROR: only 'validation' and 'training' inputs supported")
}


# ctrl + L
# clear all variables
rm(list = setdiff(ls(), lsf.str()))

strDir	= "c:/alexey_workspace/sauder/kaggle/data-science-titanic-learn/"
strData	= paste(strDir,"data/", sep = "")
strOut	= paste(strDir,"out/", sep = "")

# set current working directory
setwd(strDir)
getwd()

# read data from the input file
strFileIn	= paste(strData,"train/train.csv", sep = "")
strFileOut	= paste(strOut,"out.csv", sep = "")
strLog		= paste(strOut,"out-log.csv", sep = "")

# general env settings
bExcludeNA = TRUE

rawdata <- read.table(strFileIn, header = TRUE, sep = ",")

training	= fnSplitTrainingValidation(rawdata,"training")
summary(training)

# set explanatory variable 
y = as.factor(training$Survived)

# get rid of factors that are for sure irrelevant
namesToRemove = c("PassengerId","Survived","Name","Ticket")
training.preprocessed = fnRemoveFactors(training, namesToRemove)

training.preprocessed = fnReshuffleData(training.preprocessed,strLog);

# determine number of nFactors
nN = length(colnames(training.preprocessed))

# # get some insight into the data

# standard deviations
# sapply(training.preprocessed, sd)

pairs(cbind(y,training.preprocessed$Age,training.preprocessed$Fare), lower.panel = fnOutCorrPanel)

# plot(training.preprocessed$Age,y,xlab = "passenger age", ylab = "survived no/yes")
# boxplot(training.preprocessed$Age~y, data = training, col = c("yellow", "orange"), ylab = "passenger age")

# check if collinearity is a problem
VIFs = vif(lm(y ~ training.preprocessed$Age + training.preprocessed$Fare))

# sqrt(VIFs) > 2 indicates a problem 
bProblem = sqrt(VIFs) > 2

# fit the model: y = Pclass + Sex + Age + SibSp + Parch + Fare + Cabin + Embarked
mymodel.full = glm(y ~ .,family = binomial(link = logit), data = training.preprocessed)
summary(mymodel.full, cor = TRUE)

mymodel.null = glm(y~1,family = binomial(link = logit), data = training.preprocessed)
summary(mymodel.null, cor = TRUE)

# sink(strLog)
# sink()

# step-wise model selection
selected.model = step(mymodel.full, scope = list(lower = mymodel.null, upper = mymodel.full), direction = "backward", scale = 1, trace = 1)
summary(selected.model, cor = FALSE)
selected.model$anova
confint(selected.model)

# newdata1 <- with(training.preprocessed, data.frame(Age = mean(Age), Fare = mean(Fare), Pclass = factor(1:4)))
# newdata = with(training.preprocessed, data.frame(PassengerId = "ID", Survived = 0, Pclass = factor(1:3), Name = "NAME", Sex = factor(1:2), Age = 22, SibSp = factor(1:7), Parch = factor(1:6), Ticket = "TICKET", Fare = 7.25, Cabin = factor(1:9), Embarked = "S"))
# newdata.preprocessed = fnRemoveFactors(newdata, namesToRemove)
# newdata.preprocessed = fnReshuffleData(newdata.preprocessed,strLog);

validation	= fnSplitTrainingValidation(rawdata,"validation")

# explanatory variable from validation set 
validation.y	= as.factor(validation$Survived);

validation.preprocessed	= fnRemoveFactors(validation, namesToRemove)
validation.preprocessed	= fnReshuffleData(validation.preprocessed);

# write.csv(validation.preprocessed, file = strLog, quote = FALSE, row.names = FALSE)

# predicted values
validation.y.predict = predict.glm(selected.model, newdata = validation.preprocessed, type = "response")

# find optimal threshold by optimizing "certain" distance 
rTH = fnFindOptimalThreshold(validation.y,validation.y.predict, nSteps = 100)

# output realized vs. predicted data for validation data set
data.PredictVsRealized = data.frame(validation.y,ifelse((validation.y.predict > rTH), 1.0, 0.0))
colnames(data.PredictVsRealized) = c("y", "predicted y")
write.csv(data.PredictVsRealized, file = strFileOut, quote = FALSE, row.names = FALSE)