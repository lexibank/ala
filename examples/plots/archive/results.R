library(readr)
library(dplyr)
library(ggdist)
library(gghalves)
library(ggplot2)
library(ggrepel)
library(viridis)
library(brms)
library(tidybayes)
library(stringr)


full_data <- c()
for (model in c('grambank', 'lexibank', 'asjp', 'combined')){
  data <- read_tsv(paste("../../results/results_", model, ".tsv", sep=''))
  full_data <- rbind(full_data, data)
}

per_model<- full_data %>% group_by(Model, Run) %>% 
  filter(Family=='TOTAL') %>% 
  summarise(Score=mean(Score))

colors_vc <- c('#0c71ff', '#ca2800', '#ff28ba', '#000096')
violin_complex <- per_model %>% 
  ggplot(aes(x=reorder(Model, Score), y=Score)) +
  geom_half_point(aes(fill=Model), side="l", range_scale=.25, alpha=.5, size=0.2) +
  stat_halfeye(aes(fill=Model), adjust=2, width=0.5, color=NA, position=position_nudge(x=0.10)) +
  geom_boxplot(width=0.15, color="darkgrey") +
  coord_flip() +
  #scale_fill_viridis(discrete=TRUE, end=0.95) +
  scale_fill_discrete(colors_vc) +
  scale_y_continuous(breaks=c(60, 70, 80, 90)) +
  scale_x_discrete(label=NULL, name=NULL, breaks=NULL) +
  theme_grey(base_size=14) +
  theme(legend.position='bottom', legend.title=element_blank(),
        axis.text=element_text(size=12),
        axis.title.x=element_blank(),
        legend.text=element_text(size=13),
        )

violin_complex
ggsave("violin_complex.pdf", plot=violin_complex, dpi=300, width=1500, height=1200, units="px")

#####################
per_family <- full_data %>% group_by(Family, Model) %>%
  filter(Family != 'TOTAL') %>% 
  summarise(Score=mean(Score), Languages=mean(Languages))

FamsToLabel <- c('Salishan', 'Austronesian', 'Indo-European')

colors_sc <- c('#0c71ff', '#ca2800', '#ff28ba', '#000096', '#86e300', '#1c5951', '#20d2ff', '#20ae86', '#590000', '#65008e', '#b6005d', '#ffaa96', '#ba10c2', '#510039', '#00650c', '#0096a6', '#20aa00', '#ffaeeb', '#ff316d', '#0431ff', '#31e7ce', '#eb65ff', '#ff6d2d', '#8a2071', '#24ffa6', '#002d1c', '#7d7151', '#042441', '#28658a', '#c69aaa', '#922020', '#927186', '#599aff', '#69c66d', '#6d2d41', '#ffe779', '#a2a2c6', '#008671', '#9204d2', '#beebff', '#c6149a', '#3d10a6', '#413d59', '#b28a5d', '#ebcad2', '#004100', '#657900', '#e71486', '#00beb2', '#ce1c45', '#a671b2', '#b639ff')
per_family <- per_family %>% filter(Model=='Lexibank')
scatter <- per_family %>% 
  ggplot(aes(x=Score, y=Languages, fill=Family)) +
  geom_point(aes(size=0.7), alpha=0.9, shape=21)+
  # facet_wrap(~Model) +
  geom_label_repel(aes(label=Family), data=per_family[per_family$Family %in% FamsToLabel,],
                   max.overlaps=10, min.segment.length=unit(0, 'lines'), color="black",
                   box.padding=unit(1.1, "lines"), size=6) +
  scale_y_log10(limits=c(2.5, 950), name="Number of languages in data") +
  scale_x_continuous() +
  scale_fill_discrete(colors_sc) +
  geom_smooth(method=glm, se=TRUE, alpha=0.3, color="red", fill="#69b3a2") +
  theme(
    legend.position="none", 
    strip.text=element_text(size=16),
    axis.text=element_text(size=14),
    axis.title.x=element_blank(),
    axis.title.y=element_text(size=16), 
    )
scatter

ggsave("scatter.pdf", plot=scatter, dpi=300,  width=2000, height=1500, units="px")

#####################
scope <- full_data %>% distinct(Model, Languages, Family) %>%
  filter(Family != 'TOTAL') %>% 
  group_by(Model) %>%
  summarise(langs=sum(Languages), fams=length(unique(Family)))

scope_plot <- scope %>% 
  ggplot(aes(x=fams, y=langs, fill=Model, label=Model)) +
  geom_point(aes(size=5), shape=c(21, 23, 23, 22), alpha=0.8, position=position_dodge(width=3)) +
  scale_y_log10(limits=c(1000, 5100), breaks=c(1000, 2000, 5000)) + 
  geom_label_repel(max.overlaps=30, min.segment.length=unit(0, 'lines'), color="black",
                   box.padding=unit(0.7, "lines")) +
  theme(legend.position="none") +
  scale_fill_viridis(discrete=TRUE, begin=0.3)

scope_plot
ggsave("scope.pdf", plot=scope_plot, dpi=300,
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
ggsave("lang_acc.pdf", plot=lang_acc, dpi=300,
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
ggsave("fam_acc.pdf", plot=fam_acc, dpi=300, width=3000, height=2000, units="px")

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
  scale_y_continuous(limits=c(61, 99), breaks=c(60, 70, 80, 90, 100), 
                     name="Estimated F1-macro average") +
  scale_x_discrete(label=NULL, name=NULL, breaks=NULL) +
  theme_grey(base_size=14) +
  theme(legend.position='bottom', legend.title=element_blank())
model_comp

ggsave("model_est.pdf", plot=model_comp, dpi=300, width=3000, height=1000, units="px")


############################################
full_data <- c()
for (model in c('grambank', 'lexibank', 'combined')){
  data <- read_tsv(paste("../../results/experiments_", model, ".tsv", sep='')) %>% 
    mutate(Model=str_to_title(model))
  full_data <- rbind(full_data, data)
}

full_data <- full_data %>% 
  mutate(
    Language = str_replace(Language, 'bang1363', 'Bangime'),
    Language = str_replace(Language, 'basq1248', 'Basque'),
    Language = str_replace(Language, 'kusu1250', 'Kusunda'),
    Language = str_replace(Language, 'mapu1245', 'Mapudungun'),
  )

colors_ld <- c('#0c71ff', '#ca2800', '#ff28ba', '#000096', '#86e300', '#1c5951', '#20d2ff', '#20ae86', '#590000')
long_distance <- full_data %>% 
  filter(Family!="Unclassified") %>% 
  group_by(Family, Model, Prediction) %>% 
  summarise(Frequency=sum(Frequency)/ n_distinct(Language)) %>% 
  filter(Frequency>=10) %>% 
  group_by(Family, Model) %>% 
  slice_max(n=3, Frequency) %>% 
  ggplot(aes(y=Frequency, x=Prediction, fill=Prediction)) +
  geom_col() +
  scale_y_continuous(breaks=c(0, 50, 100), 
                     name="Frequency of prediction") +
  scale_x_discrete(name='Predicted language family') +
  scale_fill_discrete(colors_ld) +
  facet_grid(Model~Family, scales='free_x') +
  theme(
    legend.position='bottom',
    legend.title=element_blank(),
    legend.text=element_text(size=14),
    strip.text=element_text(size=18),
    axis.text.x = element_blank(),
    axis.text.y=element_text(size=12),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18)
  ) 
long_distance
ggsave("long_distance.pdf", plot=long_distance, dpi=300, width=2500, height=2500, units="px")


correct_exp <- full_data %>% 
  filter(Family!="Unclassified") %>% 
  group_by(Family, Model, Prediction) %>% 
  summarise(Frequency=sum(Frequency)/ n_distinct(Language)) %>% 
  filter(Family == Prediction) %>% 
  ggplot(aes(y=Frequency, x=reorder(Frequency, Model), fill=Model)) +
  geom_col() +
  scale_y_continuous(breaks=c(0, 50, 100), 
                     name="Frequency of prediction") +
  scale_x_discrete(name='Predicted language family') +
  scale_fill_viridis(discrete=T) +
  facet_grid(~Family, scales='free_x') +
  theme(
    legend.position='bottom',
    legend.title=element_blank(),
    axis.text.x = element_blank()
  )
correct_exp


##############
# Isolates
colors_is <- c('#0c71ff', '#ca2800', '#ff28ba', '#000096', '#86e300', '#1c5951', '#20d2ff', '#20ae86', '#590000', '#65008e', '#b6005d')
isolates <- full_data %>% 
  filter(Family=='Unclassified', Frequency>=10) %>% 
  # filter(!(Prediction %in% c('Nuclear Trans New Guinea', 'Nakh-Daghestanian', 'Dravidian', 'Chibchan', 'Nuclear-Macro-Je'))) %>% 
  # filter(!(Prediction == 'Mande' & Language == 'Kusunda')) %>% 
  select(-Family) %>% 
  group_by(Language, Model) %>% 
  slice_max(n=3, Frequency) %>% 
  ggplot(aes(y=Frequency, x=Prediction, fill=Prediction)) +
  geom_col() +
  scale_y_continuous(breaks=c(0, 50, 100), 
                     name="Frequency of prediction") +
  scale_x_discrete(name='Predicted language family') +
  scale_fill_discrete(colors_is) +
  facet_grid(Model~Language, scales='free_x') +
  theme(
    legend.position='bottom',
    legend.title=element_blank(),
    legend.text=element_text(size=14),
    strip.text=element_text(size=18),
    axis.text.x=element_blank(),
    axis.text.y=element_text(size=12),
    axis.title.x=element_text(size=18),
    axis.title.y=element_text(size=18)
  ) 
isolates
ggsave("isolates.pdf", plot=isolates, dpi=300, width=2500, height=2500, units="px")
