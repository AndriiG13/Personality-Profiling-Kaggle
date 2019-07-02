library(tidyverse)
library(tidytext)
library(stringr)
library(boot)
library(lexicon)
library(sentimentr)
library(tm)
library(modelr)
library(caret)

##LOad up the files
getwd()
##load in the transcripts
################################################################################
all_files = unzip('youtube-personality.zip', list = T)$Name

filenames = all_files[grepl("^youtube-personality/transcripts/VLOG", all_files)]

readTextFile = function(nm) {
  unzip('youtube-personality.zip',nm); # unzip the requested file
  text = readChar(nm, nchars = 1000000);        # read in the text (max 1 million characters)
  unlink(nm);                                   # delete the file (otherwise Kaggle complains)
  text                                          # return the extracted text
}

# example:  
readTextFile(filenames[1])


length(filenames)
transcripts <- matrix(data = NA, ncol = 2, nrow = length(filenames))
for(i in 1:length(filenames)){
  transcripts[i,1] <- str_extract(filenames[i], "VLOG\\d+")
  transcripts[i,2] <- readTextFile(filenames[i])
}

transcripts <- as.tibble(transcripts)
names(transcripts) <- c("ID", "TEXT")
transcripts
################################################################################


##load in  ratings
################################################################################
train_ratings <- read.csv("youtube-personality/YouTube-Personality-Personality_impression_scores_train.csv")

train_ratings <- train_ratings %>%  separate(1, into = c("ID", "EXT", "AGR","CON", "EMO", "OPN"), sep = " ", convert = T)

train_ratings <- as_tibble(train_ratings)
################################################################################



##Extracting profanity 
tokens <- transcripts %>%
  unnest_tokens(token, TEXT, token = 'words')

prof <- as_tibble(lexicon::profanity_alvarez)

profanity_count <- inner_join(tokens, prof, by = c("token" = "value")) %>% 
  count(ID, token) %>% group_by(ID) %>% summarise(n_prof = sum(n))

profanity <- left_join(train_ratings, profanity_count)

profanity$n_prof[is.na(profanity$n_prof)] <- 0


##EXtracting pronouns 
pronouns <- inner_join(tokens, pos_df_pronouns, by = c(token = "pronoun"))

pronouns <- dplyr::select(pronouns, ID, token, type)
pronouns <- pronouns %>% group_by(ID, type) %>%  summarise(count = n()) %>% spread(type, count, fill = 0)


##extracting emotions
emo <- get_sentences(transcripts$TEXT)
emo <- emotion_by(emo)

emo_wide <- emo %>% dplyr::select(element_id, emotion_type, emotion_count) %>% 
  spread(emotion_type, emotion_count)

emotions_id <- cbind(transcripts$ID, emo_wide)

emotions_sw <- inner_join(train_ratings,emotions_id[,-2], by = c(ID = "V1"))


###SENTIMENT FROM SENTIMENTR
transcripts$TEXT.sw <- removeWords(transcripts$TEXT,stopwords("english"))
sent <- get_sentences(transcripts$TEXT.sw)
sent<- sentiment_by(sent)

sent_wide <- sent %>% dplyr::select(element_id, ave_sentiment) 

sentiment_id <- cbind(transcripts$ID, sent_wide)

ave_sentiment <- inner_join(train_ratings,sentiment_id[,-2], by = c(ID = "V1") )

###load in audiovisual features
################################################################################
audio_visual <- read.table("youtube-personality/YouTube-Personality-audiovisual_features.csv", header = T)
################################################################################
train_audiovisual <- inner_join(train_ratings,audio_visual, by = c("ID" = "vlogId"))
train_audiovisual <- train_audiovisual[,-1]


################################################################################
gender <- read.table("youtube-personality/YouTube-Personality-gender.csv", header = T)
################################################################################

train_gender <- inner_join(train_ratings,gender, by = c("ID" = "vlogId"))
train_gender$gender <-  ifelse(train_gender$gender == "Male", 1,0)
train_gender$gender <- as.factor(train_gender$gender)


##making a final data set 
s_p <- inner_join(ave_sentiment, profanity)
s_p_e <- inner_join(s_p, emotions_sw )
s_p_e_av <- inner_join(s_p_e, train_audiovisual)
s_p_e_av_g <- inner_join(s_p_e_av, train_gender)
s_p_e_av_g_p <- inner_join(s_p_e_av_g, pronouns)


##Using forward and backward selection to pick the variables. Then choosing between f and b 
#selections using cross validatoin 
##PICKING EMOTION MODEL
##First fitted models with all observations and then examined cook's distance and removed
##some outliers
fitstart <- lm(EMO~1, data = s_p_e_av_g_p[-c(204,214,147),c(-1,-6,-3,-2,-4)])
fitALL <- lm(EMO~., data = s_p_e_av_g_p[-c(204,214,147),c(-1,-6,-3,-2,-4)])


##frward selection 
fit_emo <- step(fitstart, direction = "forward", scope = formula(fitALL))
fit_emo <- lm(EMO ~ n_prof + ave_sentiment + hogv.cogC + time.speaking + personal + 
                   surprise_negated, data = s_p_e_av_g_p[-c(204,214,147),c(-1,-6,-3,-2,-4)])

train_control <- trainControl(method="repeatedcv", number=10, repeats=10)

train(EMO ~ n_prof + ave_sentiment + hogv.cogC + time.speaking + personal + 
        surprise_negated, data = s_p_e_av_g_p[, c(-1, -2, -3, -6, -4)],
      trControl=train_control,
      method="lm")


###PICKING EXTROVERSION MODEL
fitstart <- lm(EXT~1, data = s_p_e_av_g_p[-c(43,78),c(-1,-6,-3,-5,-4)])
fitALL <- lm(EXT~., data = s_p_e_av_g_p[-c(43,78),c(-1,-6,-5,-3,-4)])

fit_ext <- stepAIC(fitstart, direction = "forward", scope = formula(fitALL))
summary(fit_ext)

fit_ext <- lm(EXT ~ hogv.entropy + time.speaking + sd.d.energy + 
                 mean.loc.apeak + sd.loc.apeak + gender + mean.pitch + possessive + 
                 mean.energy + mean.spec.entropy + trust_negated + joy + num.turns + 
                 mean.d.energy + sadness + anger + hogv.cogR + mean.val.apeak + 
                 mean.conf.pitch, data = s_p_e_av_g_p[-c(43,78), c(-1, -6, -3, -5, -4)])

train(EXT~hogv.entropy + time.speaking + sd.d.energy + 
        mean.loc.apeak + sd.loc.apeak + gender + mean.pitch + possessive + 
        mean.energy + mean.spec.entropy + trust_negated + joy + num.turns + 
        mean.d.energy + sadness + anger + hogv.cogR + mean.val.apeak + 
        mean.conf.pitch, data = s_p_e_av_g_p[ , c(-1, -6, -3, -5, -4)],
               trControl=train_control,
               method="lm")



##PICKING THE BEST AGR MODEL 
fitstart <- lm(AGR~1, data = s_p_e_av_g_p[-c(204, 285),c(-1,-6,-2,-5,-4)])
fitALL <- lm(AGR~., data = s_p_e_av_g_p[-c(204, 285),c(-1,-6,-5,-2,-4)])


fit_agr <- step(fitstart, direction = "forward", scope = formula(fitALL))
fit_agr <- lm(AGR ~ n_prof + ave_sentiment + gender + anger_negated + hogv.cogC + 
                trust_negated + anticipation_negated + possessive + hogv.cogR, 
                 data = s_p_e_av_g_p[-c(204, 285), c(-1,-6, -2, -5, -4)])
summary(fit_agr)

train_control <- trainControl(method="repeatedcv", number=10, repeats=10)
# train the model
train(AGR ~ n_prof + ave_sentiment + gender + anger_negated + hogv.cogC + 
                 trust_negated + anticipation_negated + possessive + hogv.cogR, data = s_p_e_av_g_p[-c(204, 285),c(-1,-6,-2,-5,-4)],
               trControl=train_control,
               method="lm")




##PICKING THE BEST CON MODEL 
fitstart <- lm(CON~1, data = s_p_e_av_g_p[-204,c(-1,-6,-2,-5,-3)])
fitALL <- lm(CON~., data = s_p_e_av_g_p[-204,c(-1,-6,-5,-2,-3)])


fit_con <- step(fitstart, direction = "both", scope = formula(fitALL))
fit_con <- glm(CON ~ time.speaking + n_prof + trust + disgust_negated + hogv.entropy + 
                   hogv.cogC + fear_negated + anger_negated + avg.len.seg + 
                   ave_sentiment + reflexive + joy_negated, 
                 data = s_p_e_av_g_p[-204, c(-1, -6, -2, -5, -3)])



train_control <- trainControl(method="repeatedcv", number=10, repeats=10)

train(CON ~ time.speaking + n_prof + trust + disgust_negated + hogv.entropy + 
        hogv.cogC + fear_negated + anger_negated + avg.len.seg + 
        ave_sentiment + reflexive + joy_negated, data = s_p_e_av_g_p[-204,c(-1,-6,-2,-5,-3)],
               trControl=train_control,
               method="lm")

##Picking the best OPN MODEL 
fitstart <- lm(OPN~1, data = s_p_e_av_g_p[-c(46,322),c(-1,-4,-2,-5,-3)])
fitALL <- lm(OPN~., data = s_p_e_av_g_p[-c(46,322),c(-1,-4,-5,-2,-3)])


fit_opn <- step(fitstart, direction = "both", scope = formula(fitALL))
fit_opn <- glm(OPN ~ hogv.median + time.speaking + sd.loc.apeak + disgust + 
                   joy + gender + mean.pitch + sd.conf.pitch + surprise + surprise_negated + 
                   sd.val.apeak + disgust_negated + anger_negated, 
                 data = s_p_e_av_g_p[-c(46,322), c(-1, -4, -2, -5, -3)])



train(OPN ~ hogv.median + time.speaking + sd.loc.apeak + disgust + 
        joy + gender + mean.pitch + sd.conf.pitch + surprise + surprise_negated + 
        sd.val.apeak + disgust_negated + anger_negated, data = s_p_e_av_g_p[-c(46,322),c(-1,-4,-2,-5,-3)],
               trControl=train_control,
               method="lm")

##make the test dataset
ave_sentiment_test <- anti_join(sentiment_id[,-2],train_ratings, by = c(V1 = "ID") )

s_e_test <- inner_join(ave_sentiment_test,emotions_id[,-2] )

profanity_test <- anti_join(profanity_count, train_ratings) %>% dplyr::select(ID, n_prof)
s_p_e_test <- left_join(s_e_test, profanity_test, by = c(V1 = "ID"))
s_p_e_test$n_prof[is.na(s_p_e_test$n_prof)] <-  0

s_p_e_a_test <- inner_join(s_p_e_test,audio_visual, by = c("V1" = "vlogId") )

s_p_e_av_g_test <- inner_join(s_p_e_a_test, gender, by = c(V1 = "vlogId"))
s_p_e_av_g_test$gender <-  ifelse(s_p_e_av_g_test$gender == "Male", 1,0)
s_p_e_av_g_test$gender <- as.factor(s_p_e_av_g_test$gender)

s_p_e_av_g_p_test <- inner_join(s_p_e_av_g_test, pronouns, by = c(V1 = "ID"))

##make the final document
test_predicted <- s_p_e_av_g_p_test %>%  add_predictions(fit_ext, var = "Extr") %>% 
  add_predictions(fit_agr, var = "Agr") %>%  add_predictions(fit_con, var = "Cons") %>% 
  add_predictions(fit_emo, var = "Emot") %>%  add_predictions(fit_opn, var = "Open")

test_predicted <- test_predicted %>% dplyr::select(V1, Extr, Agr, Cons, Emot, Open)

names(test_predicted) <- c("ID" ,  "Extr", "Agr" , "Cons", "Emot", "Open")

test_predicted

test_predicted <- test_predicted %>%  gather("type", "Expected", -ID) %>% arrange(desc(ID))

final_test <- test_predicted %>% unite(ID,ID, type, sep = "_")

write_csv(final_test,path = "predictions_final_set_no3.csv")




