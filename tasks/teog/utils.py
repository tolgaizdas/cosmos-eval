obp_6 = 80
obp_7 = 80
obp_8 = 80

weights = {
    1: 4,
    2: 4,
    3: 2,
    4: 4,
    5: 2,
    6: 2,
}


def get_exam_score(dataframe, sinav):
    def ders_basari_hesapla(dataframe, sinav):
        ders = dataframe[dataframe['sinav'] == sinav]
        dogru_sayisi = ders['dogruyanlis'].sum()
        return (dogru_sayisi / len(ders) * 100) * weights[sinav]

    turkce_basari = ders_basari_hesapla(dataframe, 1)
    print(turkce_basari)
    matematik_basari = ders_basari_hesapla(dataframe, 2)
    print(matematik_basari)
    inkilap_basari = ders_basari_hesapla(dataframe, 3)
    print(inkilap_basari)
    fen_basari = ders_basari_hesapla(dataframe, 4)
    print(fen_basari)
    ingilizce_basari = ders_basari_hesapla(dataframe, 5)
    print(ingilizce_basari)
    din_basari = ders_basari_hesapla(dataframe, 6)
    print(din_basari)

    exam_score = turkce_basari + matematik_basari + inkilap_basari + fen_basari + ingilizce_basari + din_basari
    total_score = (obp_6 + obp_7 + obp_8 + exam_score * (7 / 18)) / 2
    return total_score
