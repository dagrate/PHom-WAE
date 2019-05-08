library(TDA)
library(TDAmapper)
library(ggplot2)
library(igraph)
library(locfit)
library(ks)
library(networkD3)


BOTTLENECKDIST <- 0
BRCDE <- 1

#### open the file
IDICT <- 1
dict <- c("originalSamples.csv", "WAE.csv", "VAE.csv")


#### Rips Diagram
if (BOTTLENECKDIST == 1) {
  batch <- 500
  maxBatch <- 4000    
  maxscale <- 1 # limit of the filtration
  maxdimension <- 1 # H-dimension
  
  N <- maxBatch / batch
  IDICT <- 1
  flnm <- dict[IDICT]
  xfraudsA <- read.table( paste("", flnm, sep="") , sep=',')
  lastcolumn <- dim(xfraudsA)[2] - 1
  
  totBottleneck <- c(0,0)
  for (idict in seq(1, 2)) {
    flnm <- dict[idict + 1]
    xfraudsB <- read.table( paste("", flnm, sep="") , sep=',')
    
    n0 <- 1
    for (nbatch in seq(0, maxBatch - batch, batch)) {
      print(paste("nbatch: ", n0 + batch - 1, "/ idict: ", idict))
      print("=============")
      
      # we shuffle the data to remove any bias
      dfA <- xfraudsA[sample(nrow(xfraudsA)),]
      dfB <- xfraudsB[sample(nrow(xfraudsB)),]
      
      Diag1 <- ripsDiag(X = dfA[n0:(n0 + batch - 1),1:lastcolumn], maxdimension, maxscale, library = "GUDHI", printProgress = TRUE)
      Diag2 <- ripsDiag(X = dfB[n0:(n0 + batch - 1),1:lastcolumn], maxdimension, maxscale, library = "GUDHI", printProgress = TRUE)
      
      curBottleneck <- bottleneck(Diag1[["diagram"]], Diag2[["diagram"]])
      print(curBottleneck)
      totBottleneck[idict] <- totBottleneck[idict] + curBottleneck
      n0 <- (n0 + batch)
    }
    # we compute the average of the bottleneck distance
    totBottleneck[idict] <- totBottleneck[idict] / N
  }
  print(totBottleneck)
}


#### Barcode
if (BRCDE == 1) {
  n0 <- 1
  batch <- 1000
  maxBatch <- 2500
  maxscale <- 0.1 # limit of the filtration
  maxdimension <- 1 # H-dimension
  
  N <- maxBatch / batch
  IDICT <- 1
  flnm <- dict[IDICT]
  print(flnm)
  xfraudsA <- read.table( paste("", flnm, sep="") , sep=',')
  lastcolumn <- dim(xfraudsA)[2] - 1
  
  dfA <- xfraudsA[sample(nrow(xfraudsA)),]
  Diag1 <- ripsDiag(X = dfA[n0:(n0 + batch - 1),1:lastcolumn], maxdimension, maxscale, library = "GUDHI", printProgress = TRUE)
  plot(Diag1[["diagram"]], barcode = TRUE)
  plot(Diag1[["diagram"]], rotated=TRUE, barcode = FALSE)
}