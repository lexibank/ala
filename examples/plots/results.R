library(readr)
library(dplyr)
library(ggplot2)
library(viridis)
library(gghalves)
library(ggdist)

data <- read_tsv("lexibank_results.tsv") %>% 
  rbind(read_tsv("grambank_results.tsv")) %>% 
  rbind(read_tsv("combined_results.tsv"))

violin <- data %>% 
  ggplot(aes(y=Model, x=Family)) +
  geom_violin(aes(fill=Model)) +
  geom_boxplot(width=0.3, 
               outlier.size=1, outlier.color="black", outlier.alpha=0.3) +
  # facet_wrap(~Language, ncol=3) +
  scale_fill_viridis(discrete=TRUE, end=0.9) +
  scale_x_continuous(limits=c(52, 100), breaks=c(50, 60, 70, 80, 90, 100), 
                name="Average Family Accuracy") +
  scale_y_discrete(label=NULL, name=NULL) +
  theme_grey(base_size=14) +
  theme(legend.position='bottom', legend.title=element_blank())
violin
ggsave("intersection_violin.png", plot=violin)


violin_complex <- data %>% 
  ggplot(aes(y=Family, x=Model)) +
  geom_boxplot(aes(fill=Model), width=.2, size=0.3, outlier.shape=NA) +
  geom_half_point(aes(fill=Model),side="l", range_scale=.25, alpha=.5, size=0.1) +
  stat_halfeye(aes(fill=Model), adjust=1, width=.5, color=NA, position=position_nudge(x=.15)) +
  coord_flip() +
  scale_fill_viridis(discrete=TRUE, end=0.9) +
  scale_y_continuous(limits=c(53, 100), breaks=c(60, 70, 80, 90, 100), 
                     name="Average Family Accuracy") +
  scale_x_discrete(label=NULL, name=NULL, breaks=NULL) +
  theme_grey(base_size=14) +
  theme(legend.position='bottom', legend.title=element_blank())
violin_complex
ggsave("intersection_violin_complex.png", plot=violin_complex)


full_data <- read_tsv("lexibank_results_ind.tsv") %>% 
  rbind(read_tsv("grambank_results_ind.tsv")) %>% 
  rbind(read_tsv("combined_results.tsv"))

violin <- full_data %>% 
  ggplot(aes(y=Model, x=Family)) +
  geom_violin(aes(fill=Model)) +
  geom_boxplot(width=0.3, 
               outlier.size=1, outlier.color="black", outlier.alpha=0.3) +
  # facet_wrap(~Language, ncol=3) +
  scale_fill_viridis(discrete=TRUE, end=0.9) +
  scale_x_continuous(limits=c(52, 100), breaks=c(50, 60, 70, 80, 90, 100), 
                     name="Average Family Accuracy") +
  scale_y_discrete(label=NULL, name=NULL) +
  theme_grey(base_size=14) +
  theme(legend.position='bottom', legend.title=element_blank())
violin
ggsave("full_data_violin.png", plot=violin)


violin_complex <- full_data %>% 
  ggplot(aes(y=Family, x=Model)) +
  geom_boxplot(aes(fill=Model), width=.2, size=0.3, outlier.shape=NA) +
  geom_half_point(aes(fill=Model),side="l", range_scale=.25, alpha=.5, size=0.1) +
  stat_halfeye(aes(fill=Model), adjust=1, width=.5, color=NA, position=position_nudge(x=.15)) +
  coord_flip() +
  scale_fill_viridis(discrete=TRUE, end=0.9) +
  scale_y_continuous(limits=c(53, 100), breaks=c(60, 70, 80, 90, 100), 
                     name="Average Family Accuracy") +
  scale_x_discrete(label=NULL, name=NULL, breaks=NULL) +
  theme_grey(base_size=14) +
  theme(legend.position='bottom', legend.title=element_blank())
violin_complex
ggsave("full_data_violin_complex.png", plot=violin_complex)
