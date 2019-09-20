if (!'rcdk' %in% installed.packages()){
  install.packages("rcdk")
}

get_fingerprint <- function(smiles, type){
  mol <- rcdk::parse.smiles(smiles)[[1]]
  fps <- rcdk::get.fingerprint(mol, type)
  output <- rep(0, fps@nbit)
  output[fps@bits] <- 1
  return(output)
}

get_descriptors <- function(smiles, type){
  mol <- rcdk::parse.smiles(smiles)
  dnames <- rcdk::get.desc.names()
  descs <- rcdk::eval.desc(mol, dnames, verbose=FALSE)
  return(as.numeric(descs))
}

write_sdf <- function(smiles, out_file){
  mol <- rcdk::parse.smiles(smiles)
  rcdk::write.molecules(mol, out_file)
}