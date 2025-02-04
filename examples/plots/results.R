library(readr)
library(dplyr)
library(ggdist)
library(gghalves)
library(ggplot2)
library(ggrepel)
library(viridis)
library(brms)
library(tidybayes)


gb_all <- read_tsv("../results/results_grambank.tsv") %>%
  mutate(Model='Grambank')

lb_all <- read_tsv("../results/results_lexibank.tsv") %>%
  mutate(Model='Lexibank')

asjp_all <- read_tsv("../results/results_asjp.tsv") %>%
  mutate(Model='ASJP')

combined <- read_tsv("../results/results_combined.tsv") %>%
  mutate(Model='Combined')

full_data <- rbind(gb_all, lb_all, asjp_all, combined)

per_model<- full_data %>% group_by(Model, Run) %>% 
  filter(Family=='TOTAL') %>% 
  summarise(Score=mean(Score))

violin_complex <- per_model %>% 
  ggplot(aes(x=reorder(Model, Score), y=Score)) +
  geom_half_point(aes(fill=Model), side="l", range_scale=.25, alpha=.5, size=0.2) +
  stat_halfeye(aes(fill=Model), adjust=1, width=0.75, color=NA, position=position_nudge(x=0.06)) +
  geom_boxplot(width=0.1, color="grey", alpha=0.5) +
  coord_flip() +
  scale_fill_viridis(discrete=TRUE, end=0.95) +
  scale_y_continuous(limits=c(50, 99.5), breaks=c(60, 70, 80, 90, 100), 
                     name="F1-macro average") +
  scale_x_discrete(label=NULL, name=NULL, breaks=NULL) +
  theme_grey(base_size=14) +
  theme(legend.position='bottom', legend.title=element_blank())

violin_complex
ggsave("violin_complex.png", plot=violin_complex, dpi=300, width=2000, height=1500, units="px")

#####################
per_family <- full_data %>% group_by(Family, Model) %>%
  filter(Family != 'TOTAL') %>% 
  summarise(Score=mean(Score), Languages=mean(Languages)) %>% 
  group_by(Family) %>% count() %>% arrange(-n)

FamsToLabel <- c('Nuclear-Macro-Je', 'Austronesian', 'Indo-European', 'Koiarian')

scatter <-  per_family %>% 
  ggplot(aes(x=Score, y=Languages, fill=Family)) +
  geom_point(aes(size=1), shape=21) +
  geom_label_repel(aes(label=Family), data=per_family[per_family$Family %in% FamsToLabel,],
                   max.overlaps=10, min.segment.length=unit(0, 'lines'), color="black",
                   box.padding=unit(1.5, "lines"), size=6) +
  scale_y_log10(limits=c(3, 1000)) + 
  scale_fill_viridis(discrete=TRUE, option="D", begin=0.2) +
  # geom_smooth(method=lm , color="red", fill="#69b3a2", se=FALSE) +
  theme(
    legend.position="none", 
    strip.text=element_text(size=20),
    axis.text=element_text(size=16),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18),
    ) +
  facet_wrap(~Model)
scatter

ggsave("scatter.png", plot=scatter, dpi=300,  width=3000, height=2000, units="px")

#####################
scope <- full_data %>% distinct(Model, Languages, Family) %>%
  filter(family != 'TOTAL') %>% 
  group_by(Model) %>%
  summarise(langs=sum(Languages), fams=length(unique(Family)))
scope

scope_plot <- scope %>% 
  ggplot(aes(x=fams, y=langs, fill=Model, label=Model)) +
  geom_point(aes(size=5), shape=c(21, 23, 23, 22), alpha=0.8, position=position_dodge(width=3)) +
  scale_y_log10(limits=c(800, 5000), breaks=c(1000, 2000, 5000)) + 
  geom_label_repel(max.overlaps=30, min.segment.length=unit(0, 'lines'), color="black",
                   box.padding=unit(0.7, "lines")) +
  theme(legend.position="none") +
  scale_fill_viridis(discrete=TRUE, begin=0.3)

scope_plot
ggsave("scope.png", plot=scope_plot, dpi=300,
       width=3000, height=2000, units="px")

lang_acc <- scope %>% left_join(per_model) %>% group_by(Model) %>%
  summarise(Score=mean(Score), langs=mean(langs), fams=mean(fams)) %>% 
  ggplot(aes(x=Score, y=langs, fill=Model, label=Model)) +
  geom_point(aes(size=5), shape=c(21, 23, 23, 22), alpha=0.8) +
  scale_y_log10(limits=c(800, 5000), breaks=c(1000, 2000, 5000)) + 
  geom_label_repel(max.overlaps=30, min.segment.length=unit(0, 'lines'), color="black",
                   box.padding=unit(0.7, "lines")) +
  theme(legend.position="none") +
  scale_fill_viridis(discrete=TRUE, begin=0.3)

lang_acc
ggsave("lang_acc.png", plot=lang_acc, dpi=300,
       width=3000, height=2000, units="px")

fam_acc <- scope %>% left_join(per_model) %>% group_by(Model) %>%
  summarise(Score=mean(Score), langs=mean(langs), fams=mean(fams)) %>% 
  ggplot(aes(x=Score, y=fams, fill=Model, label=Model)) +
  geom_point(aes(size=5), shape=c(21, 23, 23, 22), alpha=0.8) +
  geom_label_repel(max.overlaps=30, min.segment.length=unit(0, 'lines'), color="black",
                   box.padding=unit(0.7, "lines")) +
  theme(legend.position="none") +
  scale_fill_viridis(discrete=TRUE, begin=0.3)

fam_acc
ggsave("fam_acc.png", plot=fam_acc, dpi=300, width=3000, height=2000, units="px")

################################
### Statistical model       ####
################################
mod_robust <- brm(
  bf(Score ~ Model, sigma ~ Model),
  family=student,
  data=per_model,
  cores=4,
  iter=2000,
  warmup=1000
)

predictions <- add_epred_draws(newdata=per_model, object=mod_robust)

model_comp <- predictions %>% 
  ggplot(aes(x=reorder(Model, .epred), y=.epred)) +
  stat_halfeye(aes(fill=Model), adjust=5) +
  coord_flip() +
  scale_fill_viridis(discrete=TRUE, end=0.95) +
  scale_y_continuous(limits=c(60, 99.5), breaks=c(60, 70, 80, 90, 100), 
                     name="Estimated Family Score") +
  scale_x_discrete(label=NULL, name=NULL, breaks=NULL) +
  theme_grey(base_size=14) +
  theme(legend.position='bottom', legend.title=element_blank())
model_comp

ggsave("model_est.png", plot=model_comp, dpi=300, width=3000, height=2000, units="px")
