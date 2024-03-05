library(readr)
library(dplyr)
library(forcats)
library(ggdist)
library(gghalves)
library(ggplot2)
library(ggrepel)
library(viridis)

gb_all <- read_tsv("../results/results_grambank_no_detailed.tsv") %>%
  mutate(Model='Grambank_all')

lb_all <- read_tsv("../results/results_lexibank_no_detailed.tsv") %>%
  mutate(Model='Lexibank_all')

gb_intersec <- read_tsv("../results/results_grambank_intersec_detailed.tsv") %>%
  mutate(Model='Grambank_intersec')

lb_intersec <- read_tsv("../results/results_lexibank_intersec_detailed.tsv") %>%
  mutate(Model='Lexibank_intersec')

asjp_all <- read_tsv("../results/results_asjp_no_detailed.tsv") %>%
  mutate(Model='ASJP_all')

asjp_intersec <- read_tsv("../results/results_asjp_intersec_detailed.tsv") %>%
  mutate(Model='ASJP_intersec')

combined <- read_tsv("../results/results_combined_no_detailed.tsv") %>%
  mutate(Model='LexiGram_combined')


full_data <- rbind(gb_all, lb_all, gb_intersec, lb_intersec, asjp_all, asjp_intersec, combined)

per_model<- full_data %>% group_by(Model, Run) %>% 
  summarise(Accuracy = mean(Accuracy))

violin_complex <- per_model %>% 
  ggplot(aes(x=reorder(Model, Accuracy), y=Accuracy)) +
  geom_half_point(aes(fill=Model),side="l", range_scale=.25, alpha=.5, size=0.2) +
  stat_summary(fun = "mean",
               geom = "crossbar", 
               width = 0.5,
               colour = "red") +
  stat_halfeye(aes(fill=Model), adjust=1, width=.6, color=NA, position=position_nudge(x=.15)) +
  coord_flip() +
  scale_fill_viridis(discrete=TRUE, end=0.9) +
  scale_y_continuous(limits=c(60, 99.5), breaks=c(60, 70, 80, 90, 100), 
                     name="Average Family Accuracy") +
  scale_x_discrete(label=NULL, name=NULL, breaks=NULL) +
  theme_grey(base_size=14) +
  theme(legend.position='bottom', legend.title=element_blank())

violin_complex
ggsave("violin_complex.png", plot=violin_complex, dpi=300,
       width=2000, height=1500, units="px")


#####################
per_family <- full_data %>% group_by(Family, Model) %>%
  summarise(Accuracy = mean(Accuracy), Languages= mean(Languages))

scatter <-  per_family %>% 
  ggplot(aes(x=Accuracy, y=Languages, fill=Family, label=Family)) +
  geom_point(aes(size=1), shape=21) +
  geom_label_repel(max.overlaps=30, min.segment.length=unit(0, 'lines'), color="black",
                   box.padding = unit(2, "lines")) +
  scale_y_log10(limits = c(3, 500)) + 
  scale_fill_viridis(discrete=TRUE, option="D", begin=0.3) +
  # geom_smooth(method=lm , color="red", fill="#69b3a2", se=FALSE) +
  theme(legend.position="none") +
  facet_wrap(~Model)
scatter
# marginal boxplot
# scatter <- ggMarginal(scatter, type="boxplot")
ggsave("scatter.png", plot=scatter, dpi=300,
       width=3000, height=2000, units="px")

#####################
scope <- per_family %>% group_by(Model) %>%
  summarise(langs=sum(Languages), fams=length(unique(Family)))

scope_plot <- scope %>% 
  ggplot(aes(x=fams, y=langs, fill=Model, label=Model)) +
  geom_point(aes(size=5), shape=c(21, 23, 23, 22, 22, 24, 22), alpha=0.8, position=position_dodge(width=3)) +
  scale_y_log10(limits = c(800, 5000), breaks=c(1000, 2000, 5000)) + 
  geom_label_repel(max.overlaps=30, min.segment.length=unit(0, 'lines'), color="black",
                   box.padding = unit(0.7, "lines")) +
  theme(legend.position="none") +
  scale_fill_viridis(discrete=TRUE, begin=0.3)
scope_plot
ggsave("scope.png", plot=scope_plot, dpi=300,
       width=3000, height=2000, units="px")


lang_acc <- scope %>% left_join(per_model) %>% group_by(Model) %>%
  summarise(Accuracy=mean(Accuracy), langs=mean(langs), fams=mean(fams)) %>% 
ggplot(aes(x=Accuracy, y=langs, fill=Model, label=Model)) +
  geom_point(aes(size=5), shape=c(21, 23, 23, 22, 22, 24, 22), alpha=0.8) +
  scale_y_log10(limits = c(800, 5000), breaks=c(1000, 2000, 5000)) + 
  geom_label_repel(max.overlaps=30, min.segment.length=unit(0, 'lines'), color="black",
                   box.padding = unit(0.7, "lines")) +
  theme(legend.position="none") +
  scale_fill_viridis(discrete=TRUE, begin=0.3)
lang_acc
ggsave("lang_acc.png", plot=lang_acc, dpi=300,
       width=3000, height=2000, units="px")

fam_acc <- scope %>% left_join(per_model) %>% group_by(Model) %>%
  summarise(Accuracy=mean(Accuracy), langs=mean(langs), fams=mean(fams)) %>% 
  ggplot(aes(x=Accuracy, y=fams, fill=Model, label=Model)) +
  geom_point(aes(size=5), shape=c(21, 23, 23, 22, 22, 24, 22), alpha=0.8) +
  geom_label_repel(max.overlaps=30, min.segment.length=unit(0, 'lines'), color="black",
                   box.padding = unit(0.7, "lines")) +
  theme(legend.position="none") +
  scale_fill_viridis(discrete=TRUE, begin=0.3)
fam_acc
ggsave("fam_acc.png", plot=fam_acc, dpi=300,
       width=3000, height=2000, units="px")
