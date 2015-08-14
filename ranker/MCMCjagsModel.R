library(rube)
library(R2jags)
library(plyr)
library(reshape2)

get_name <- function(filename){
  return(unlist(strsplit(as.character(filename), '\\.'))[1])
}

gold_data <- read.table('/Users/cmaclell/Projects/pairwise/ranker/gold-ratings.csv', header=T, sep=",", quote="\"", check.names=F)
gold_data <- melt(gold_data, variable.name="judge_id", value.name="rating_value")
gold_data <- gold_data[complete.cases(gold_data),]

individual_data <- read.table('/Users/cmaclell/Projects/pairwise/ranker/individual.csv', header=T, sep=",", quote="\"", check.names=F)
individual_data$item <- unlist(lapply(individual_data$item_name, get_name))

pairwise_data <- read.table('/Users/cmaclell/Projects/pairwise/ranker/pairwise.csv', header=T, sep=",", quote="\"", check.names=F)
pairwise_data$left <- unlist(lapply(pairwise_data$left_item_name, get_name))
pairwise_data$right <- unlist(lapply(pairwise_data$right_item_name, get_name))

original_items = unique(c(as.character(gold_data$item), 
                          as.character(individual_data$item), 
                          as.character(pairwise_data$left), 
                          as.character(pairwise_data$right)))
new_item_ids = seq(1,length(original_items), 1)
estimates = data.frame(new_item_ids, original_items)
colnames(estimates) = c('id', 'item')

gold_data$new_item_id <- mapvalues(gold_data$item, from=estimates$item, to=estimates$id)
individual_data$new_item_id <- mapvalues(individual_data$item, from=estimates$item, to=estimates$id)
pairwise_data$new_left_id <- mapvalues(pairwise_data$left, from=estimates$item, to=estimates$id)
pairwise_data$new_right_id <- mapvalues(pairwise_data$right, from=estimates$item, to=estimates$id)

individual_data$new_judge_id <- mapvalues(individual_data$judge_id, from = unique(individual_data$judge_id), to = seq(1, length(unique(individual_data$judge_id))))
pairwise_data$new_judge_id <- mapvalues(pairwise_data$judge_id, from = unique(pairwise_data$judge_id), to = seq(1, length(unique(pairwise_data$judge_id))))

model("/Users/cmaclell/Projects/pairwise/ranker/jointMCMC.rube")

# MCMC initial values
abinit = function(data, extra) {
  lst = list(
    alpha0=c(-2, -1, 0, 1, 2, 3),
    MEAN=runif(1,1,7))
  return(lst)
}

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
  NITEMS=length(new_ids),
  NJUDGES=length(unique(individual_data$new_judge_id)),
  NPJUDGES=length(unique(pairwise_data$new_judge_id))
)

abinit(rube_data)

rube("/Users/cmaclell/Projects/pairwise/ranker/jointMCMC.rube", 
     rube_data, abinit,)
model = rube("/Users/cmaclell/Projects/pairwise/ranker/jointMCMC.rube", 
                   rube_data, abinit, "*",
                   n.burnin=1000, n.thin=15, n.iter=3000, bin=30, ignore='all')
p3(model, PIprob=0.95)
summary(model, limit=1) #1060

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
          n.burnin=1000, n.thin=15, n.iter=3000, bin=30, ignore='all')
p3(gold_model)
summary(gold_model)


ids$rube_est = gold_model$mean$trueCreativity
mle <- read.table('/Users/cmaclell/Projects/pairwise/ranker/item_estimates.csv', header=T, sep=",", quote="\"", check.names=F)
colnames(mle) = c('id', 'name', 'mle_est', 'mle_conf')

compare = join(ids, mle)
plot(compare$rube_est, compare$mle_est)
cor(compare$rube_est, compare$mle_est, method="spearman")

