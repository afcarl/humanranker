library(rube)
library(R2jags)
library(plyr)

individual_data <- read.table('/Users/cmaclell/Projects/pairwise/ranker/individual.csv', header=T, sep=",", quote="\"", check.names=F)
pairwise_data <- read.table('/Users/cmaclell/Projects/pairwise/ranker/pairwise.csv', header=T, sep=",", quote="\"", check.names=F)

original_ids = unique(c(pairwise_data$left_item_id, pairwise_data$right_item_id, individual_data$item_id))
new_ids = seq(1,length(original_ids),1)
ids = data.frame(original_ids, new_ids)
colnames(ids) <- c('id', 'rube_id')

individual_data$new_judge_id <- mapvalues(individual_data$judge_id, from = unique(individual_data$judge_id), to = seq(1, length(unique(individual_data$judge_id))))
individual_data$new_item_id <- mapvalues(individual_data$item_id, from = original_ids, to = new_ids)

pairwise_data$new_left_id <- mapvalues(pairwise_data$left_item_id, from = original_ids, to = new_ids)
pairwise_data$new_right_id <- mapvalues(pairwise_data$right_item_id, from = original_ids, to = new_ids)
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
m1 = rube("/Users/cmaclell/Projects/pairwise/ranker/jointMCMC.rube", 
                   rube_data, abinit, "*",
                   n.burnin=2000, n.thin=15, n.iter=10000, bin=30, ignore='all')
p3(m1, PIprob=0.95)
summary(m1, limit=1) #1060

ids$rube_est = m1$mean$trueCreativity
mle <- read.table('/Users/cmaclell/Projects/pairwise/ranker/item_estimates.csv', header=T, sep=",", quote="\"", check.names=F)
colnames(mle) = c('id', 'name', 'mle_est', 'mle_conf')

compare = join(ids, mle)
plot(compare$rube_est, compare$mle_est)
cor(compare$rube_est, compare$mle_est, method="spearman")
