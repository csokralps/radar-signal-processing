# radar-signal-processing
MATLAB ile geliştirilmiş özgün radar sinyal işleme çalışmalar

OFDM haberleşme sinyalinin radar olarak kullanıldığı (ISAC) bir sistemin MATLAB simülasyonu. Haberleşme için gönderilen QPSK sembollerinden kanal kestirilir, menzil-Doppler haritası çıkarılır ve hedefler 2B CA-CFAR ile tespit edilir. Ardından aynı işleme zinciri sabit nokta aritmetiğine taşınarak bit genişliğinin tespit performansına etkisi incelenir.

## Sistem parametreleri

| Parametre | Değer |
|---|---|
| Taşıyıcı frekansı | 5.8 GHz |
| Bant genişliği | 16 MHz |
| Alt taşıyıcı / OFDM sembolü | 256 / 64 |
| Döngüsel önek | %25 (sembol süresi 20 µs) |
| Menzil çözünürlüğü | 9.375 m |
| Hız çözünürlüğü | ≈ 20.2 m/s |
| Teorik entegrasyon kazancı | 42.1 dB |
| CFAR penceresi | 13 × 7 (76 eğitim hücresi) |

## Neler yapılıyor

1. **Kanal kestirimi ve entegrasyon kazancı:** Menzil-Doppler haritasında hedefin doğru hücrede bulunduğu ve ölçülen kazancın teorik değerle uyuştuğu kontrol edilir.
2. **CA-CFAR doğrulaması:** Dikdörtgen ve Hamming pencereleri için yanlış alarm olasılığı (Pfa), ayrı kalibrasyon ve doğrulama kareleriyle ölçülür.
3. **Güçlü hedef yanında zayıf hedef tespiti:** Zayıf hedef, güçlü hedefin koruma bölgesine, CFAR eğitim bölgesine, pencere dışına ve yakın Doppler hücresine yerleştirilerek güç farkına göre tespit olasılığı (Pd) çıkarılır. Zayıf hedef yokken aynı bölgede oluşan tespitler ayrıca ölçülerek yan lob kaynaklı yanlış tespitler ayrıştırılır.
4. **Sabit nokta analizi:** İşleme zinciri (pencereleme, IFFT, FFT, güç, CFAR eşiği) blok seviyesinde nicemlenir. 8–20 bit aralığında Pd, Pfa, SQNR, ölü bölge ve doyma oranları hesaplanır; her format için eşik ayrıca kalibre edilir.
5. **Nicemleme kaybı:** Aynı Monte Carlo denemeleri kayan noktada da işlenerek Pd kaybının ne kadarının nicemlemeden, ne kadarının gürültü ve yan loblardan kaynaklandığı ayrıştırılır.

## Sonuçlar

![Şekil 1](figures/sekil1_cfar_ve_konum.png)
*Menzil-Doppler haritası, CFAR Pfa doğrulaması ve hedef konumuna göre zayıf hedef tespiti*

![Şekil 2](figures/sekil2_sabit_nokta.png)
*Bit genişliğine göre sabit nokta Pd ve Pfa*

![Şekil 3](figures/sekil3_ana_matris.png)
*Bit genişliği × pencere × güç farkı deney matrisi*

![Şekil 4](figures/sekil4_nicemleme_kaybi.png)
*Kayan noktaya göre nicemleme kaynaklı Pd kaybı*

## Çalıştırma

```matlab
ofdm_isac_radar_sim
```
