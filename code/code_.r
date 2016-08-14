# http://www.ats.ucla.edu/stat/r/dae/logit.htm

# source("c:/alexey_workspace/sauder/kaggle/data-science-titanic-learn/code/code.R")
library(car)

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
		throw()
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

rawdata <- read.table(strFileIn, header = TRUE, sep = ",")

training	= fnSplitTrainingValidation(rawdata)

summary(training)

# get rid of factors that are for sure irrelevant
namesToRemove = c("PassengerId","Name")
training.preprocessed = fnRemoveFactors(training, namesToRemove)

# standard deviations
sapply(training.preprocessed, sd)

# determine number of nFactors
nN = length(colnames(training.preprocessed)) - 1

bExcludeNA = TRUE

y = as.factor(training.preprocessed$Survived)

x3 = as.factor(training.preprocessed$Pclass)

x5 = as.factor(training.preprocessed$Sex)

x6.raw = training.preprocessed$Age
x6 = fnScaleData(x6.raw, bExcludeNA)

x7 = as.factor(training.preprocessed$SibSp)

x8 = as.factor(training.preprocessed$Parch)

x10.raw = training.preprocessed$Fare
x10 = fnScaleData(x10.raw, bExcludeNA)

x11 = as.factor(substr(training.preprocessed$Cabin, 1, 1))
# write.csv(x11, file = strFileOut, quote = FALSE, row.names = FALSE)

x12 = as.factor(training.preprocessed$Embarked)

pairs(cbind(y,x6,x10), lower.panel = fnOutCorrPanel)

# # get some insight into the data
# plot(x6,y,xlab = "passenger age", ylab = "survived no/yes")
# boxplot(x6~y, data = training, col = c("yellow", "orange"), ylab = "passenger age")

# check if collinearity is a problem
VIFs = vif(lm(y ~ x6 + x10))
# sqrt(VIFs) > 2 indicates a problem 
bProblem = sqrt(VIFs) > 2

newdata0 = data.frame(x3,x5,x6,x7,x8,x10,x11);

# fit the model
mymodel.fit = glm(y ~ x3 + x5 + x6 + x7 + x8 + x10 + x11,family = binomial(link = logit), data = training)
summary(mymodel.fit, cor = TRUE)
# sink(strFileOut)
# sink()

mymodel.null = lm(y~1)
# full=lm(Price~., data=Housing)

# step-wise model selection
selected.model = step(mymodel.fit, scope = list(lower = mymodel.null, upper = mymodel.fit), direction = "backward", scale = 1, trace = 1)
summary(selected.model, cor = FALSE)
selected.model$anova

# confint(selected.model)

# newdata1 <- with(training.preprocessed, data.frame(Age = mean(Age), Fare = mean(Fare), Pclass = factor(1:4)))

newdata2 <- with(training.preprocessed, data.frame(PassengerId = "AAAAAAA", Survived = 1, Pclass = factor(1), Name = "BBBBBBBB", Sex = "male", Age = mean(Age,na.rm = TRUE), SibSp = factor(1), Parch = factor(1), Ticket = "CCCCCCC", Fare = mean(Fare), Cabin = "C", Embarked = factor(1)))
newdata2.preprocessed = fnRemoveFactors(newdata2, namesToRemove)

# predict values
newdata2.predict = predict(selected.model, newdata = newdata2.preprocessed, interval = 'prediction')
# predict(selected.model, newdata = fnScaleData(50.0,,mean(x6, na.rm = bExcludeNA),sd(x6, na.rm = bExcludeNA)))