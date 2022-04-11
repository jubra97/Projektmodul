a = load("16_0Hidden\auswertung_on_sps.mat")
b = load("12_2Hidden\auswertung_on_sps.mat")

c = load("16_0Hidden\auswertung_with_ads.mat")
d = load("12_2Hidden\auswertung_with_ads.mat")

figure
plot(a.data(3, :))
hold on
plot(b.data(3, :))
plot(c.data(3, :))
plot(d.data(3, :))

plot(a.data(2, :))
plot(b.data(2, :))
plot(c.data(2, :))
plot(d.data(2, :))

figure
plot(a.data(4, :))
hold on
plot(b.data(4, :))
plot(c.data(4, :))
plot(d.data(4, :))