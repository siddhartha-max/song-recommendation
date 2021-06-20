{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.metrics.pairwise import sigmoid_kernel"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "songs = pd.read_excel('data.xls')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Video ID</th>\n",
       "      <th>Songs</th>\n",
       "      <th>Artists</th>\n",
       "      <th>singer</th>\n",
       "      <th>music</th>\n",
       "      <th>lyrics</th>\n",
       "      <th>Channel</th>\n",
       "      <th>View Count</th>\n",
       "      <th>Like Count</th>\n",
       "      <th>Dislike Count</th>\n",
       "      <th>Comment Count</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>kckDWrICC4s</td>\n",
       "      <td>NAZAR LAG JAYEGI</td>\n",
       "      <td>Kamal Raja,Millind Gaba</td>\n",
       "      <td>MilindGaba &amp; KamalRaja</td>\n",
       "      <td>MusicMG</td>\n",
       "      <td>Ikrar,MillindGaba</td>\n",
       "      <td>T-Series</td>\n",
       "      <td>177575589</td>\n",
       "      <td>1612878.0</td>\n",
       "      <td>106200.0</td>\n",
       "      <td>50701.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>88LgZ-cf_P4</td>\n",
       "      <td>Suit Suit</td>\n",
       "      <td>NaN</td>\n",
       "      <td>GuruRandhawaFt.Arjun</td>\n",
       "      <td>Intense</td>\n",
       "      <td>GuruRandhawaandArjun</td>\n",
       "      <td>T-Series</td>\n",
       "      <td>155165181</td>\n",
       "      <td>550783.0</td>\n",
       "      <td>52198.0</td>\n",
       "      <td>18380.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>HV3-5ivdY88</td>\n",
       "      <td>Bholeynath</td>\n",
       "      <td>Millind Gaba, Ikka, Pallavi Gaba</td>\n",
       "      <td>MillindGaba,Ikka,PallaviGaba</td>\n",
       "      <td>MillindGaba</td>\n",
       "      <td>Ikka</td>\n",
       "      <td>T-Series</td>\n",
       "      <td>65083886</td>\n",
       "      <td>479157.0</td>\n",
       "      <td>28577.0</td>\n",
       "      <td>11609.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>zyVaheF55SM</td>\n",
       "      <td>Dil Dooba</td>\n",
       "      <td>Aishwarya Rai, Akshaye Kumar</td>\n",
       "      <td>SonuNigam,ShreyaGhoshal</td>\n",
       "      <td>RamSampat</td>\n",
       "      <td>Sameer</td>\n",
       "      <td>T-Series</td>\n",
       "      <td>46336027</td>\n",
       "      <td>260590.0</td>\n",
       "      <td>9634.0</td>\n",
       "      <td>6970.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>jRszQSQZu1o</td>\n",
       "      <td>Suit Suit</td>\n",
       "      <td>NaN</td>\n",
       "      <td>GuruRandhawaFt.Arjun</td>\n",
       "      <td>Intense</td>\n",
       "      <td>GuruRandhawaandArjun</td>\n",
       "      <td>T-Series</td>\n",
       "      <td>24514920</td>\n",
       "      <td>89280.0</td>\n",
       "      <td>10673.0</td>\n",
       "      <td>2230.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      Video ID             Songs                            Artists  \\\n",
       "0  kckDWrICC4s  NAZAR LAG JAYEGI            Kamal Raja,Millind Gaba   \n",
       "1  88LgZ-cf_P4         Suit Suit                                NaN   \n",
       "2  HV3-5ivdY88       Bholeynath   Millind Gaba, Ikka, Pallavi Gaba    \n",
       "3  zyVaheF55SM         Dil Dooba       Aishwarya Rai, Akshaye Kumar   \n",
       "4  jRszQSQZu1o         Suit Suit                                NaN   \n",
       "\n",
       "                         singer        music                lyrics   Channel  \\\n",
       "0        MilindGaba & KamalRaja      MusicMG     Ikrar,MillindGaba  T-Series   \n",
       "1          GuruRandhawaFt.Arjun      Intense  GuruRandhawaandArjun  T-Series   \n",
       "2  MillindGaba,Ikka,PallaviGaba  MillindGaba                  Ikka  T-Series   \n",
       "3       SonuNigam,ShreyaGhoshal    RamSampat                Sameer  T-Series   \n",
       "4          GuruRandhawaFt.Arjun      Intense  GuruRandhawaandArjun  T-Series   \n",
       "\n",
       "   View Count  Like Count  Dislike Count  Comment Count  \n",
       "0   177575589   1612878.0       106200.0        50701.0  \n",
       "1   155165181    550783.0        52198.0        18380.0  \n",
       "2    65083886    479157.0        28577.0        11609.0  \n",
       "3    46336027    260590.0         9634.0         6970.0  \n",
       "4    24514920     89280.0        10673.0         2230.0  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "songs.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_null=songs.dropna(how='any')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(718, 11)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_null.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Video ID         0\n",
       "Songs            0\n",
       "Artists          0\n",
       "singer           0\n",
       "music            0\n",
       "lyrics           0\n",
       "Channel          0\n",
       "View Count       0\n",
       "Like Count       0\n",
       "Dislike Count    0\n",
       "Comment Count    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_null.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Songs</th>\n",
       "      <th>Artists</th>\n",
       "      <th>singer</th>\n",
       "      <th>lyrics</th>\n",
       "      <th>music</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>NAZAR LAG JAYEGI</td>\n",
       "      <td>Kamal Raja,Millind Gaba</td>\n",
       "      <td>MilindGaba &amp; KamalRaja</td>\n",
       "      <td>Ikrar,MillindGaba</td>\n",
       "      <td>MusicMG</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Bholeynath</td>\n",
       "      <td>Millind Gaba, Ikka, Pallavi Gaba</td>\n",
       "      <td>MillindGaba,Ikka,PallaviGaba</td>\n",
       "      <td>Ikka</td>\n",
       "      <td>MillindGaba</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Dil Dooba</td>\n",
       "      <td>Aishwarya Rai, Akshaye Kumar</td>\n",
       "      <td>SonuNigam,ShreyaGhoshal</td>\n",
       "      <td>Sameer</td>\n",
       "      <td>RamSampat</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Laal Dupatta</td>\n",
       "      <td>Mika Singh and Anupama Raag</td>\n",
       "      <td>MikaSingh&amp;AnupamaRaag</td>\n",
       "      <td>AnupamaRaag</td>\n",
       "      <td>AnupamaRaag</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Deedar De</td>\n",
       "      <td>Abhishek Bacchan</td>\n",
       "      <td>SunidhiChauhan,Krishna</td>\n",
       "      <td>PanchhiJalonvi</td>\n",
       "      <td>VishalDadlani,ShekharRavjiani</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "              Songs                            Artists  \\\n",
       "0  NAZAR LAG JAYEGI            Kamal Raja,Millind Gaba   \n",
       "2       Bholeynath   Millind Gaba, Ikka, Pallavi Gaba    \n",
       "3         Dil Dooba       Aishwarya Rai, Akshaye Kumar   \n",
       "5      Laal Dupatta        Mika Singh and Anupama Raag   \n",
       "6         Deedar De                   Abhishek Bacchan   \n",
       "\n",
       "                         singer             lyrics  \\\n",
       "0        MilindGaba & KamalRaja  Ikrar,MillindGaba   \n",
       "2  MillindGaba,Ikka,PallaviGaba               Ikka   \n",
       "3       SonuNigam,ShreyaGhoshal             Sameer   \n",
       "5         MikaSingh&AnupamaRaag        AnupamaRaag   \n",
       "6        SunidhiChauhan,Krishna     PanchhiJalonvi   \n",
       "\n",
       "                           music  \n",
       "0                        MusicMG  \n",
       "2                    MillindGaba  \n",
       "3                      RamSampat  \n",
       "5                    AnupamaRaag  \n",
       "6  VishalDadlani,ShekharRavjiani  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_rec=df_null[['Songs','Artists','singer','lyrics','music']]\n",
    "df_rec.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "df_rec.to_csv('df_rec.csv',index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "Songs_rec = pd.read_csv('df_rec.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "Songs_rec ['Songs'] = Songs_rec['Songs'].apply(lambda x:x.lower())\n",
    "tfidf = TfidfVectorizer(min_df=1,max_features =None,strip_accents= \"unicode\",analyzer='word',token_pattern=r'\\w{1,}',ngram_range=(1,1),stop_words='english')\n",
    "tf_matrix = tfidf.fit_transform(Songs_rec['Songs'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "sgm = sigmoid_kernel(tf_matrix,tf_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "indices = pd.Series(Songs_rec.index,index=Songs_rec['Songs']).drop_duplicates()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def recommend(name,sig=sgm):\n",
    "    idx = indices[name]\n",
    "    songs_scores = list(enumerate(sig[idx]))\n",
    "    songs_scores = sorted(songs_scores,key = lambda x:x[1],reverse=True)\n",
    "    top_10_similar_songs = songs_scores[1:11]  \n",
    "    songs_indices = [i[0] for i in top_10_similar_songs]\n",
    "    return Songs_rec['Songs'].iloc[songs_indices]   "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
