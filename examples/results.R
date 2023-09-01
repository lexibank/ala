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
  geom_boxplot(width=0.2, 
               outlier.size=1, outlier.color="black", outlier.alpha=0.3) +
  # facet_wrap(~Language, ncol=3) +
  scale_fill_viridis(discrete=TRUE, end=0.7) +
  scale_x_continuous(limits=c(50, 100), breaks=c(50, 60, 70, 80, 85, 90, 95, 100), 
                name="Average Family Accuracy") +
  scale_y_discrete(label=NULL, name=NULL) +
  theme_grey(base_size=11) +
  theme(legend.position='bottom', legend.title=element_blank())
violin
ggsave("violin_simple.png", plot=violin)


violin_complex <- data %>% 
  ggplot(aes(y=Family, x=Model)) +
  geom_boxplot(aes(fill=Model), width=.2, size=0.3, outlier.shape=NA) +
  geom_half_point(aes(fill=Model),side="l", range_scale=.25, alpha=.5, size=0.1) +
  stat_halfeye(aes(fill=Model), adjust=1, width=.5, color=NA, position=position_nudge(x=.15)) +
  coord_flip() +
  scale_fill_viridis(discrete=TRUE, end=0.7) +
  scale_y_continuous(limits=c(50, 100), breaks=c(50, 60, 70, 80, 85, 90, 95, 100), 
                     name="Average Family Accuracy") +
  scale_x_discrete(label=NULL, name=NULL) +
  theme_grey(base_size=11) +
  theme(legend.position='bottom', legend.title=element_blank())
violin_complex
ggsave("violin_complex.png", plot=violin_complex)

bar <- data %>% 
  group_by(Model) %>% 
  summarise(avg_family=mean(Family),
            sd=sd(Family)) %>% 
  ggplot(aes(y=avg_family, x=Model)) +
  geom_bar(aes(fill=Model), stat="identity", width=0.5) +
  geom_errorbar(aes(ymin=avg_family-sd, ymax=avg_family+sd),
                width=0.3)+
  scale_fill_viridis(discrete=TRUE, end=0.7) +
  scale_x_discrete(label=NULL, name=NULL) +
  scale_y_continuous(name="Average Family Accuracy") +
  theme_grey(base_size=11) +
  theme(legend.position='bottom', legend.title=element_blank())
bar
ggsave("bar.png", plot=bar)

