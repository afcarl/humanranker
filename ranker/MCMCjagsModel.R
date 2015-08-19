library(rube)
library(R2jags)
library(plyr)
library(reshape2)

get_name <- function(filename){
  return(unlist(strsplit(as.character(filename), '\\.'))[1])
}

gold_data <- read.table('/Users/cmaclell/Projects/pairwise/ranker/gold_data_complete.csv', header=T, sep=",", quote="\"", check.names=F)
gold_data <- subset(gold_data, select=c('ideaID','joel value', 'joel novelty', 'angela value', 'angela novelty'))

# novelty
gold_data$`1` <- gold_data$`joel novelty`
gold_data$`2` <- gold_data$`angela novelty`

# combined
#gold_data$`1` <- gold_data$`joel value` * gold_data$`joel novelty`
#gold_data$`2` <- gold_data$`angela value` * gold_data$`angela novelty`

gold_data <- subset(gold_data, select=c('ideaID','1', '2'))
colnames(gold_data) <- c('item', '1', '2')
gold_data <- melt(gold_data, variable.name="judge_id", value.name="rating_value")
gold_data <- gold_data[complete.cases(gold_data),]
gold_data <- gold_data[gold_data$rating_value!=0,]
gold_data$item <- as.character(gold_data$item)

individual_data <- read.table('/Users/cmaclell/Projects/pairwise/ranker/individual.csv', header=T, sep=",", quote="\"", check.names=F)
individual_data$item <- as.character(unlist(lapply(individual_data$item_name, get_name)))

pairwise_data <- read.table('/Users/cmaclell/Projects/pairwise/ranker/pairwise.csv', header=T, sep=",", quote="\"", check.names=F)
pairwise_data$left <- as.character(unlist(lapply(pairwise_data$left_item_name, get_name)))
pairwise_data$right <- as.character(unlist(lapply(pairwise_data$right_item_name, get_name)))

original_items = unique(c(gold_data$item, 
                          individual_data$item, 
                          pairwise_data$left, 
                          pairwise_data$right))
new_item_ids = seq(1,length(original_items), 1)
estimates = data.frame(new_item_ids, original_items)
colnames(estimates) = c('id', 'item')

gold_data$new_item_id <- as.numeric(mapvalues(gold_data$item, from=estimates$item, to=estimates$id))
individual_data$new_item_id <- as.numeric(mapvalues(individual_data$item, from=estimates$item, to=estimates$id))
pairwise_data$new_left_id <- as.numeric(mapvalues(pairwise_data$left, from=estimates$item, to=estimates$id))
pairwise_data$new_right_id <- as.numeric(mapvalues(pairwise_data$right, from=estimates$item, to=estimates$id))

individual_data$new_judge_id <- mapvalues(individual_data$judge_id, from = unique(individual_data$judge_id), to = seq(1, length(unique(individual_data$judge_id))))
pairwise_data$new_judge_id <- mapvalues(pairwise_data$judge_id, from = unique(pairwise_data$judge_id), to = seq(1, length(unique(pairwise_data$judge_id))))

ind_pair = unique(c(individual_data$new_item_id, pairwise_data$new_left_id, pairwise_data$new_right_id))
gold_processed <- gold_data
#gold_processed <- gold_processed[gold_processed$new_item_id %in% ind_pair,]
gold_processed$judge_id <- as.numeric(gold_processed$judge_id)
colnames(gold_processed) <- c('item_name', 'judge_id', 'rating_value', 'item_id')
write.csv(gold_processed, file = "/Users/cmaclell/Projects/pairwise/ranker/gold-processed.csv", row.names=FALSE, quote=FALSE)

individual_processed <- individual_data
#individual_processed <- individual_processed[individual_processed$new_item_id %in% unique(gold_processed$item_id),]
individual_processed <- subset(individual_processed, select=c('item','rating_value', 'judge_id', 'new_item_id'))
colnames(individual_processed) <- c('item_name', 'rating_value', 'judge_id', 'item_id')
write.csv(individual_processed, file = "/Users/cmaclell/Projects/pairwise/ranker/individual-processed.csv", row.names=FALSE, quote=FALSE)

pairwise_processed <- pairwise_data
#pairwise_processed <- pairwise_processed[pairwise_processed$new_left_id %in% unique(gold_processed$item_id),]
#pairwise_processed <- pairwise_processed[pairwise_processed$new_right_id %in% unique(gold_processed$item_id),]
pairwise_processed <- subset(pairwise_processed, select=c('left','right', 'new_left_id', 'new_right_id', 'judge_id', 'rating_value'))
colnames(pairwise_processed) <- c('left_item_name', 'right_item_name', 'left_item_id', 'right_item_id', 'judge_id', 'rating_value')
write.csv(pairwise_processed, file = "/Users/cmaclell/Projects/pairwise/ranker/pairwise-processed.csv", row.names=FALSE, quote=FALSE)

model("/Users/cmaclell/Projects/pairwise/ranker/jointMCMC.rube")

# MCMC initial values
abinit = function(data, extra) {
  lst = list(
    alpha0=c(-2, -1, 0, 1, 2, 3),
    MEAN=runif(1,1,7))
  return(lst)
}

rube_data = list( 
  prating=pairwise_data$rating_value,
  pjudge=pairwise_data$new_judge_id,
  item=individual_data$new_item_id,
  left=pairwise_data$new_left_id,
  right=pairwise_data$new_right_id,
  NRATINGS=0,
  NPRATINGS=length(pairwise_data$rating_value),
  NITEMS=length(estimates$id),
  NJUDGES=0,
  NPJUDGES=length(unique(pairwise_data$new_judge_id))
)

rube_data = list( 
  rating=individual_data$rating_value,
  prating=pairwise_data$rating_value,
  judge=individual_data$new_judge_id,
  pjudge=pairwise_data$new_judge_id,
  item=individual_data$new_item_id,
  left=pairwise_data$new_left_id,
  right=pairwise_data$new_right_id,
  NRATINGS=length(individual_data$rating_value),
  NPRATINGS=length(pairwise_data$rating_value),
  NITEMS=length(estimates$id),
  NJUDGES=length(unique(individual_data$new_judge_id)),
  NPJUDGES=length(unique(pairwise_data$new_judge_id))
)

abinit(rube_data)

rube("/Users/cmaclell/Projects/pairwise/ranker/jointMCMC.rube", 
     rube_data, abinit,)
model = rube("/Users/cmaclell/Projects/pairwise/ranker/jointMCMC.rube", 
                   rube_data, abinit, "*",
                   n.burnin=4000, n.thin=10, n.iter=10000, bin=30, ignore='all')
p3(model, PIprob=0.95)
summary(model, limit=1) #2533

estimates$turk_est = model$mean$trueCreativity

################ GOLD DATA #####################
gold_rube_data = list( 
  rating=gold_data$rating_value,
  judge=gold_data$judge_id,
  item=gold_data$new_item_id,
  NRATINGS=length(gold_data$rating_value),
  NPRATINGS=0,
  NITEMS=length(new_item_ids),
  NJUDGES=length(unique(gold_data$judge_id)),
  NPJUDGES=0
)

rube("/Users/cmaclell/Projects/pairwise/ranker/jointMCMC.rube", 
     gold_rube_data, abinit,)

gold_model = rube("/Users/cmaclell/Projects/pairwise/ranker/jointMCMC.rube", 
          gold_rube_data, abinit, "*",
          n.burnin=3000, n.thin=15, n.iter=10000, bin=30, ignore='all')
p3(gold_model)
summary(gold_model) #7301

estimates$gold_est = gold_model$mean$trueCreativity


# drop elements not in turk data
turk_items = unique(c(individual_data$item, pairwise_data$right, pairwise_data$left))
estimates = estimates[estimates$item %in% turk_items, ]

# drop elements not in gold data
gold_items = unique(gold_data$item)
estimates = estimates[estimates$item %in% gold_items, ]

ggplot(estimates) + geom_point(aes(x=gold_est, y=turk_est)) + geom_smooth(aes(x=gold_est, y=turk_est)) +
  xlab("Estimated Expert Novelty") +
  ylab("Estimated Ind+Pair Turk Novelty") +
  ggtitle("Expert vs. Ind+Pair Turk Estimates (spearman = 0.69) ")
cor(estimates$gold_est, estimates$turk_est, method="spearman")

mle <- read.table('/Users/cmaclell/Projects/pairwise/ranker/item_estimates.csv', header=T, sep=",", quote="\"", check.names=F)
colnames(mle) = c('id', 'name', 'mle_est', 'mle_conf')
mle$item <- as.character(unlist(lapply(mle$name, get_name)))
mle <- subset(mle, select=c('item','mle_est'))
#mle$id <- as.numeric(mapvalues(mle$item, from=estimates$item, to=estimates$id))

compare = join(estimates, mle)
ggplot(compare) + geom_point(aes(x=turk_est, y=mle_est)) + geom_smooth(aes(x=turk_est, y=mle_est)) +
  xlab("MCMC Turk Estimate") +
  ylab("MLE Estimate") +
  ggtitle("MCMC turk estimate vs. MLE turk estimates")
cor(compare$turk_est, compare$mle_est, method="spearman")

ggplot(compare) + geom_point(aes(x=gold_est, y=mle_est)) + geom_smooth(aes(x=gold_est, y=mle_est)) +
  xlab("MCMC Gold Estimate") +
  ylab("MLE Estimate") +
  ggtitle("MCMC gold estimate vs. MLE turk estimate")
cor(compare$gold_est, compare$mle_est, method="spearman")

