import glasbey
import seaborn as sns

sns.set_theme()

colors = glasbey.create_palette(palette_size=4, colorblind_safe=True)
print('4:', colors)

colors = glasbey.create_palette(palette_size=10, colorblind_safe=True)
print('10:', colors)

colors = glasbey.create_palette(palette_size=52, colorblind_safe=True)
print('52:', colors)
