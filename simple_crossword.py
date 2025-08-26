#!/usr/bin/env python3
"""
Simple Crossword Generator
A lightweight crossword generator that doesn't require external LLM services.
"""

import argparse
import json
import random
import re
import sys
import time
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

def load_italian_dictionary(filename: str = 'data/dizionario_italiano.json') -> Dict[str, str]:
    """Load Italian dictionary from JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Combine all words from all categories into a single dictionary
        combined_clues = {}
        for category_name, category_data in data['categories'].items():
            if 'words' in category_data:
                combined_clues.update(category_data['words'])
        
        print(f"Loaded {len(combined_clues)} Italian clues from {filename}")
        return combined_clues
    
    except FileNotFoundError:
        print(f"Warning: Dictionary file {filename} not found. Using fallback clues.")
        return FALLBACK_CLUES
    except Exception as e:
        print(f"Error loading dictionary: {e}. Using fallback clues.")
        return FALLBACK_CLUES

# Fallback clues in case the JSON file is not available
FALLBACK_CLUES = {
    'AGO': 'Strumento acuto da cucito',
    'ALA': 'Parte dell\'uccello per volare',
    'ALI': 'Due organi del volo',
    'AMA': 'Prova affetto per qualcuno',
    'AMO': 'Uncino per pescare',
    'API': 'Insetti che fanno il miele',
    'ARA': 'Verbo: coltiva la terra',
    'ARC': 'Mezzo cerchio',
    'ARS': 'Arte in latino',
    'ASI': 'Strumento per affilare',
    'BAR': 'Locale per bere caffè',
    'BLU': 'Colore del cielo sereno',
    'BOA': 'Serpente che stritola',
    'BUE': 'Bovino maschio castrato',
    'CAN': 'Cane in forma abbreviata',
    'CHE': 'Pronome interrogativo',
    'CHI': 'Domanda sulla persona',
    'CIA': 'Forma di congedo',
    'CON': 'Preposizione di compagnia',
    'COR': 'Cuore in forma poetica',
    'CUD': 'Parte posteriore',
    'CUI': 'Pronome relativo',
    'DAL': 'Preposizione articolata',
    'DEL': 'Preposizione con articolo',
    'DUE': 'Numero dopo l\'uno',
    'ECC': 'Eccetera abbreviato',
    'EGO': 'L\'io in latino',
    'ERA': 'Periodo storico',
    'ERE': 'Erbe al plurale',
    'EST': 'Punto cardinale',
    'ETA': 'Lettera greca',
    'ETÀ': 'Anni di vita',
    'EVA': 'Prima donna biblica',
    'FBI': 'Polizia federale USA',
    'FRA': 'Preposizione temporale',
    'GAG': 'Battuta comica',
    'GAS': 'Stato della materia',
    'GAY': 'Omosessuale',
    'GEL': 'Sostanza gelatinosa',
    'GIÀ': 'Avverbio di tempo',
    'GIU': 'Verso il basso',
    'GOL': 'Rete nel calcio',
    'GPS': 'Sistema di navigazione',
    'GRU': 'Uccello dal lungo collo',
    'HIT': 'Successo musicale',
    'HOT': 'Caldo in inglese',
    'ICE': 'Ghiaccio in inglese',
    'IDE': 'Idea in forma breve',
    'IER': 'Ieri abbreviato',
    'IMO': 'Secondo me (internet)',
    'IMP': 'Diavoletto',
    'IVA': 'Imposta sul valore aggiunto',
    'LAC': 'Lago in latino',
    'LAI': 'Canto popolare',
    'LED': 'Diodo luminoso',
    'LEI': 'Pronome femminile',
    'LUI': 'Pronome maschile',
    'LUX': 'Unità di illuminamento',
    'MAC': 'Impermeabile',
    'MAI': 'Avverbio di negazione',
    'MAN': 'Uomo in inglese',
    'MAX': 'Massimo abbreviato',
    'MIX': 'Miscuglio',
    'MOB': 'Folla inferocita',
    'NOI': 'Pronome prima persona plurale',
    'NON': 'Avverbio di negazione',
    'OCA': 'Volatile acquatico stupido',
    'ODE': 'Componimento poetico',
    'OHI': 'Esclamazione di dolore',
    'OIL': 'Petrolio in inglese',
    'ORA': 'Momento temporale',
    'ORE': 'Plurale di ora',
    'ORO': 'Metallo prezioso giallo',
    'OSO': 'Che osa molto',
    'PAD': 'Tavoletta elettronica',
    'PAN': 'Pane abbreviato',
    'PAR': 'Uguale, pari',
    'PAX': 'Pace in latino',
    'PER': 'Preposizione di scopo',
    'PIÙ': 'Avverbio di quantità',
    'POI': 'Avverbio di tempo successivo',
    'PRO': 'A favore di',
    'PUB': 'Locale per bere birra',
    'QUA': 'Avverbio di luogo',
    'RAG': 'Ragazzo abbreviato',
    'RAI': 'Televisione pubblica italiana',
    'RAM': 'Memoria del computer',
    'RAP': 'Genere musicale',
    'RAY': 'Raggio in inglese',
    'RED': 'Rosso in inglese',
    'REI': 'Colpevoli',
    'REX': 'Re in latino',
    'RIU': 'Ruscello sardo',
    'ROM': 'Popolazione nomade',
    'RUM': 'Liquore dei Caraibi',
    'SEI': 'Numero dopo il cinque',
    'SET': 'Insieme di oggetti',
    'SEX': 'Sesso',
    'SIA': 'Congiunzione',
    'SIR': 'Signore in inglese',
    'SIS': 'Sistema abbreviato',
    'SKI': 'Attrezzo per sciare',
    'SOS': 'Segnale di soccorso',
    'SPA': 'Centro benessere',
    'SUB': 'Palombaro',
    'SUD': 'Meridione',
    'SUN': 'Sole in inglese',
    'TAO': 'Filosofia orientale',
    'TAR': 'Catrame',
    'TAX': 'Tassa',
    'TEA': 'Tè in inglese',
    'TEN': 'Dieci in inglese',
    'TIC': 'Suono dell\'orologio',
    'TIP': 'Consiglio',
    'TON': 'Unità di peso',
    'TOP': 'La parte più alta',
    'TRA': 'Preposizione',
    'TRE': 'Numero dopo il due',
    'TUB': 'Vasca da bagno',
    'TUT': 'Faraone egizio',
    'UFO': 'Oggetto volante non identificato',
    'UNO': 'Primo numero naturale',
    'USA': 'Stati Uniti d\'America',
    'USB': 'Porta del computer',
    'UVA': 'Frutto della vite',
    'VAI': 'Imperativo di andare',
    'VAN': 'Furgone',
    'VIA': 'Strada urbana',
    'VIP': 'Persona molto importante',
    'VOI': 'Pronome seconda persona plurale',
    'WEB': 'Rete internet',
    'WIN': 'Vincere in inglese',
    'YES': 'Sì in inglese',
    'YIN': 'Principio femminile cinese',
    'ZEN': 'Filosofia orientale',
    'ZOO': 'Giardino zoologico',
    
    # Animali - massicciamente espanso
    'GATTO': 'Felino domestico che fa le fusa',
    'CANE': 'Il migliore amico dell\'uomo',
    'MICIO': 'Gatto in modo affettuoso',
    'CAGNOLINO': 'Piccolo cane da compagnia',
    'CAVALLO': 'Animale che galoppa e nitrisce',
    'LEONE': 'Il re della savana africana',
    'LEONESSA': 'Femmina del re della giungla',
    'PESCE': 'Nuota sott\'acqua con le pinne',
    'PESCIOLINO': 'Piccolo abitante dell\'acqua',
    'UCCELLO': 'Animale che vola nel cielo',
    'UCCELLINO': 'Piccolo volatile cinguettante',
    'MUCCA': 'Animale che produce latte',
    'MUCCHE': 'Bovine al pascolo',
    'PECORA': 'Animale che dà la lana',
    'PECORE': 'Gregge lanoso',
    'AGNELLO': 'Piccolo della pecora',
    'CONIGLIO': 'Roditore dalle orecchie lunghe',
    'CONIGLIETTO': 'Cucciolo di coniglio',
    'ELEFANTE': 'Pachiderma con la proboscide',
    'ELEFANTINO': 'Piccolo pachiderma',
    'TOPO': 'Piccolo roditore grigio',
    'TOPOLINO': 'Famoso topo dei cartoni',
    'ORSO': 'Plantigrado del bosco',
    'ORSETTO': 'Piccolo orso',
    'ORSACCHIOTTO': 'Peluche a forma di orso',
    'LUPO': 'Canide selvatico che ulula',
    'LUPA': 'Femmina del lupo',
    'LUPACCHIOTTO': 'Cucciolo di lupo',
    'TIGRE': 'Felino a strisce arancioni',
    'TIGROTTO': 'Cucciolo di tigre',
    'AQUILA': 'Rapace maestoso',
    'AQUILOTTO': 'Piccolo dell\'aquila',
    'COLOMBA': 'Uccello simbolo di pace',
    'COLOMBO': 'Maschio della colomba',
    'PICCIONE': 'Uccello urbano grigio',
    'GALLINA': 'Volatile del pollaio',
    'GALLO': 'Maschio della gallina',
    'PULCINO': 'Piccolo della gallina',
    'MAIALE': 'Suino della fattoria',
    'MAIALINO': 'Piccolo suino',
    'PORCELLO': 'Giovane maiale',
    'CAPRA': 'Ruminante che scala le rocce',
    'CAPRONE': 'Maschio della capra',
    'CAPRETTO': 'Piccolo della capra',
    'ASINO': 'Equino che raglia',
    'ASINA': 'Femmina dell\'asino',
    'ASINELLO': 'Piccolo asino',
    'VOLPE': 'Furba abitante del bosco',
    'VOLPACCHIOTTO': 'Cucciolo di volpe',
    'CERVO': 'Ungulato con i palchi',
    'CERVA': 'Femmina del cervo',
    'CERBIATTO': 'Piccolo cervo',
    'RANA': 'Anfibio che gracida',
    'RANOCCHIA': 'Rana femmina',
    'GIRINO': 'Larva della rana',
    'ROSPO': 'Anfibio dalla pelle rugosa',
    'SALAMANDRA': 'Anfibio dalla coda lunga',
    'FARFALLA': 'Insetto colorato che vola',
    'FARFALLINE': 'Piccole farfalle',
    'BRUCO': 'Larva della farfalla',
    'CRISALIDE': 'Stadio di trasformazione',
    'APE': 'Insetto che fa il miele',
    'APEINA': 'Regina delle api',
    'VESPA': 'Insetto che punge',
    'CALABRONE': 'Vespa di grandi dimensioni',
    'MOSCA': 'Insetto fastidioso',
    'MOSCERINO': 'Piccola mosca',
    'ZANZARA': 'Insetto che punge e succhia',
    'RAGNO': 'Aracnide tessitore',
    'RAGNETTO': 'Piccolo ragno',
    'TARANTOLA': 'Grosso ragno peloso',
    'SCORPIONE': 'Aracnide con la coda',
    'SERPENTE': 'Rettile che striscia',
    'SERPE': 'Serpente',
    'BISCIA': 'Serpente d\'acqua',
    'VIPERA': 'Serpente velenoso',
    'LUCERTOLA': 'Piccolo rettile verde',
    'GECO': 'Lucertola che cammina sui muri',
    'IGUANA': 'Grosso rettile tropicale',
    'COCCODRILLO': 'Rettile acquatico temibile',
    'ALLIGATORE': 'Coccodrillo americano',
    'TARTARUGA': 'Rettile con il guscio',
    'TESTUGGINE': 'Tartaruga di terra',
    'DELFINO': 'Mammifero marino intelligente',
    'BALENA': 'Gigante dei mari',
    'SQUALO': 'Predatore degli oceani',
    'POLPO': 'Mollusco a otto tentacoli',
    'POLIPO': 'Piccolo polpo',
    'SEPPIA': 'Mollusco che spruzza inchiostro',
    'CALAMARO': 'Mollusco con i tentacoli',
    'MEDUSA': 'Animale gelatinoso del mare',
    'STELLA': 'Stella marina',
    'RICCIO': 'Echinoderma spinoso del mare',
    'GRANCHIO': 'Crostaceo con le chele',
    'GAMBERO': 'Crostaceo dalle lunghe antenne',
    'ARAGOSTA': 'Crostaceo pregiato',
    'LUMACA': 'Mollusco con la casetta',
    'CHIOCCIOLA': 'Lumaca con guscio a spirale',
    'COCCINELLA': 'Insetto rosso a puntini',
    'SCARABEO': 'Insetto con le elitre',
    'GRILLO': 'Insetto che canta di notte',
    'CAVALLETTA': 'Insetto che salta',
    'LIBELLULA': 'Insetto con quattro ali',
    'FORMICA': 'Insetto molto laborioso',
    'FORMICHIERE': 'Animale che mangia formiche',
    'TERMITE': 'Insetto che rode il legno',
    'PIPISTRELLO': 'Mammifero che vola di notte',
    'SCOIATTOLO': 'Roditore della foresta',
    'RICCIO': 'Mammifero spinoso',
    'TALPA': 'Mammifero che scava',
    'DONNOLA': 'Piccolo mammifero furbo',
    'FAINA': 'Mammifero simile alla martora',
    'MARTORA': 'Carnivoro del bosco',
    'PUZZOLA': 'Mammifero dal cattivo odore',
    'LONTRA': 'Mammifero acquatico',
    'CASTORO': 'Roditore che costruisce dighe',
    'PORCOSPINO': 'Roditore spinoso',
    'ISTRICE': 'Mammifero dagli aculei',
    'CINGHIALE': 'Suino selvatico',
    'DAINO': 'Cervide macchiettato',
    'CAPRIOLO': 'Piccolo cervide',
    'ALCE': 'Grande cervide del nord',
    'RENNA': 'Cervide delle regioni artiche',
    'CARIBÙ': 'Renna nordamericana',
    'BISONTE': 'Bovino selvatico americano',
    'BUFALO': 'Bovino selvatico',
    'ZEBU': 'Bovino con la gobba',
    'YAK': 'Bovino tibetano',
    'LAMA': 'Camelide sudamericano',
    'ALPACA': 'Camelide dalla lana pregiata',
    'CAMMELLO': 'Animale del deserto',
    'DROMEDARIO': 'Cammello a una gobba',
    'ZEBRA': 'Equino a strisce',
    'RINOCERONTE': 'Pachiderma dal corno',
    'IPPOPOTAMO': 'Grosso mammifero acquatico',
    'GIRAFFA': 'Animale dal collo lunghissimo',
    'ANTILOPE': 'Gazzella africana',
    'GAZZELLA': 'Antilope veloce',
    'IMPALA': 'Antilope saltante',
    'GNU': 'Antilope dalla barba',
    'SCIMMIA': 'Primate arboricola',
    'SCIMMIETTA': 'Piccola scimmia',
    'GORILLA': 'Grande scimmia antropomorfa',
    'SCIMPANZÉ': 'Scimmia molto intelligente',
    'ORANGUTAN': 'Scimmia rossa del Borneo',
    'BABBUINO': 'Scimmia del muso allungato',
    'LEMURE': 'Primate del Madagascar',
    'KOALA': 'Marsupiale australiano',
    'CANGURO': 'Marsupiale che salta',
    'OPOSSUM': 'Marsupiale americano',
    'ORNITORINCO': 'Mammifero che depone uova',
    'ECHIDNA': 'Mammifero spinoso australiano',
    'PANDA': 'Orso bianco e nero cinese',
    'LEMMING': 'Piccolo roditore artico',
    'CRICETO': 'Piccolo roditore domestico',
    'GERBILLO': 'Roditore del deserto',
    'CINCILLÀ': 'Roditore dalla pelliccia morbida',
    'NUTRIA': 'Roditore acquatico',
    'MARMOTTA': 'Roditore che va in letargo',
    'PRAIRIE': 'Cane delle praterie',
    
    # Famiglia e persone
    'MADRE': 'Genitore femminile',
    'PADRE': 'Genitore maschile',
    'FIGLIO': 'Discendente maschio',
    'FIGLIA': 'Discendente femmina',
    'NONNO': 'Padre del padre',
    'NONNA': 'Madre della madre',
    'ZIO': 'Fratello del genitore',
    'ZIA': 'Sorella del genitore',
    'MARITO': 'Sposo',
    'MOGLIE': 'Sposa',
    'FRATELLO': 'Figlio maschio degli stessi genitori',
    'SORELLA': 'Figlia femmina degli stessi genitori',
    
    # Corpo umano
    'TESTA': 'Parte superiore del corpo',
    'OCCHIO': 'Organo della vista',
    'NASO': 'Organo dell\'olfatto',
    'BOCCA': 'Si usa per mangiare',
    'ORECCHIO': 'Organo dell\'udito',
    'MANO': 'Estremità del braccio',
    'PIEDE': 'Estremità della gamba',
    'BRACCIO': 'Arto superiore',
    'GAMBA': 'Arto inferiore',
    'CUORE': 'Organo che pompa il sangue',
    'CERVELLO': 'Organo del pensiero',
    
    # Casa e oggetti - molto espanso
    'CASA': 'Dolce dimora dove si abita',
    'TAVOLO': 'Mobile da pranzo con quattro gambe',
    'SEDIA': 'Mobile con schienale per sedersi',
    'LETTO': 'Mobile morbido per dormire',
    'PORTA': 'Ingresso che si apre e chiude',
    'FINESTRA': 'Apertura per far entrare la luce',
    'CUCINA': 'Stanza dove si prepara da mangiare',
    'BAGNO': 'Stanza per l\'igiene personale',
    'CAMERA': 'Stanza da letto privata',
    'GIARDINO': 'Spazio verde con fiori e piante',
    'AUTOMOBILE': 'Mezzo di trasporto a motore',
    'AUTO': 'Veicolo a quattro ruote',
    'BICICLETTA': 'Mezzo ecologico a pedali',
    'BICI': 'Due ruote senza motore',
    'LIBRO': 'Si sfoglia e si legge',
    'TELEFONO': 'Dispositivo per telefonare',
    'TETTO': 'Copertura della casa',
    'SCALA': 'Serve per salire e scendere',
    'MURO': 'Parete divisoria',
    'PAVIMENTO': 'Superficie su cui si cammina',
    'SOFFITTO': 'Parte alta della stanza',
    'ARMADIO': 'Mobile per i vestiti',
    'SPECCHIO': 'Riflette l\'immagine',
    'OROLOGIO': 'Segna le ore',
    'LAMPADINA': 'Fonte di luce artificiale',
    'CANDELA': 'Luce con la fiamma',
    'CHIAVE': 'Apre la serratura',
    'VASO': 'Contenitore per i fiori',
    'QUADRO': 'Opera d\'arte appesa al muro',
    'TENDA': 'Copre la finestra',
    'CUSCINO': 'Morbido sostegno per la testa',
    'COPERTA': 'Tessuto che scalda nel letto',
    'MATERASSO': 'Base morbida del letto',
    'FRIGORIFERO': 'Elettrodomestico che raffredda',
    'FRIGO': 'Conserva i cibi al freddo',
    'FORNO': 'Cuoce i cibi con il calore',
    'LAVASTOVIGLIE': 'Lava piatti automaticamente',
    'LAVATRICE': 'Lava i panni automaticamente',
    'TELEVISORE': 'Schermo per guardare programmi',
    'RADIO': 'Diffonde musica e notizie',
    'COMPUTER': 'Macchina elettronica intelligente',
    'MOUSE': 'Puntatore del computer',
    'TASTIERA': 'Ha tutti i tasti per scrivere',
    
    # Cibo - molto espanso
    'PANE': 'Alimento base fatto con la farina',
    'PASTA': 'Specialità italiana con sugo',
    'PIZZA': 'Piatto rotondo tipico di Napoli',
    'FORMAGGIO': 'Latticino stagionato',
    'CARNE': 'Proteina di origine animale',
    'VERDURA': 'Ortaggio verde e salutare',
    'FRUTTA': 'Dolce prodotto degli alberi',
    'ACQUA': 'Liquido trasparente e vitale',
    'VINO': 'Bevanda alcolica dell\'uva',
    'LATTE': 'Liquido bianco e nutriente',
    'UOVO': 'Si rompe per fare la frittata',
    'PESCE': 'Nuota nel mare ed è proteico',
    'RISO': 'Cereale bianco in chicchi',
    'ZUCCHERO': 'Dolcifica bevande e dolci',
    'SALE': 'Cristalli bianchi che insaporiscono',
    'MIELE': 'Dolce prodotto delle api',
    'BURRO': 'Grasso giallo da spalmare',
    'OLIO': 'Liquido per condire',
    'ACETO': 'Condimento acido',
    'POMODORO': 'Ortaggio rosso da salsa',
    'PATATA': 'Tubero che si cuoce',
    'CAROTA': 'Radice arancione',
    'CIPOLLA': 'Bulbo che fa piangere',
    'AGLIO': 'Bulbo profumato',
    'BASILICO': 'Erba aromatica verde',
    'PREZZEMOLO': 'Erba per guarnire',
    'LIMONE': 'Agrume giallo e acido',
    'ARANCIA': 'Agrume dolce e succoso',
    'MELA': 'Frutto rosso o verde',
    'PERA': 'Frutto a forma di goccia',
    'BANANA': 'Frutto giallo tropicale',
    'UVA': 'Grappolo di acini dolci',
    'FRAGOLA': 'Frutto rosso con i semini',
    'PESCA': 'Frutto vellutato estivo',
    'CILIEGIA': 'Piccolo frutto rosso',
    'PRUGNA': 'Frutto viola o giallo',
    'CAFFÈ': 'Bevanda scura energizzante',
    'TÈ': 'Bevanda calda in foglie',
    'CIOCCOLATO': 'Dolce marrone del cacao',
    'GELATO': 'Dolce freddo cremoso',
    'TORTA': 'Dolce per le feste',
    'BISCOTTO': 'Dolce secco da tè',
    'CARAMELLA': 'Dolcino colorato',
    
    # Natura - molto espanso
    'SOLE': 'Stella che illumina il giorno',
    'LUNA': 'Satellite che brilla di notte',
    'MARE': 'Distesa azzurra e salata',
    'MONTAGNA': 'Alta elevazione rocciosa',
    'FIUME': 'Acqua dolce che scorre',
    'ALBERO': 'Pianta alta con tronco e rami',
    'FIORE': 'Parte colorata e profumata',
    'ERBA': 'Verde tappeto del prato',
    'CIELO': 'Azzurra volta celeste',
    'NUVOLA': 'Bianca massa nel cielo',
    'PIOGGIA': 'Gocce d\'acqua dal cielo',
    'VENTO': 'Aria che soffia e muove',
    'FUOCO': 'Fiamma calda e luminosa',
    'TERRA': 'Pianeta azzurro dove viviamo',
    'PIETRA': 'Roccia dura del suolo',
    'SPIAGGIA': 'Riva sabbiosa del mare',
    'SABBIA': 'Granellini dorati',
    'ROCCIA': 'Massa di pietra dura',
    'COLLINA': 'Piccola elevazione verde',
    'VALLE': 'Depressione tra i monti',
    'LAGO': 'Specchio d\'acqua dolce',
    'ISOLA': 'Terra circondata dal mare',
    'BOSCO': 'Insieme di alberi',
    'FORESTA': 'Grande bosco selvaggio',
    'PRATO': 'Distesa verde di erba',
    'CAMPO': 'Terreno coltivato',
    'GIARDINO': 'Spazio verde curato',
    'PARCO': 'Area verde pubblica',
    'FOGLIA': 'Verde parte dell\'albero',
    'RAMO': 'Braccio dell\'albero',
    'RADICE': 'Parte sotterranea della pianta',
    'SEME': 'Origine di nuova pianta',
    'FRUTTO': 'Prodotto dolce dell\'albero',
    'STELLA': 'Punto di luce nel cielo',
    'PIANETA': 'Corpo celeste che orbita',
    'NEVE': 'Cristalli bianchi dal cielo',
    'GHIACCIO': 'Acqua solidificata dal freddo',
    'FULMINE': 'Scarica elettrica nel cielo',
    'TUONO': 'Boato che segue il lampo',
    'ARCOBALENO': 'Ponte colorato dopo la pioggia',
    
    # Colori - espanso
    'ROSSO': 'Colore del sangue e del fuoco',
    'BLU': 'Colore profondo del mare',
    'AZZURRO': 'Colore chiaro del cielo',
    'VERDE': 'Colore dell\'erba e foglie',
    'GIALLO': 'Colore brillante del sole',
    'NERO': 'Colore dell\'oscurità',
    'BIANCO': 'Colore puro della neve',
    'ROSA': 'Colore tenue e delicato',
    'VIOLA': 'Colore dell\'ametista e lavanda',
    'ARANCIONE': 'Colore dell\'agrume',
    'MARRONE': 'Colore del legno e terra',
    'GRIGIO': 'Colore delle nuvole',
    'ORO': 'Metallo prezioso giallo',
    'ARGENTO': 'Metallo lucido e bianco',
    'BRONZO': 'Lega metallica ramata',
    
    # Tempo e date
    'ANNO': 'Dodici mesi',
    'MESE': 'Parte dell\'anno',
    'GIORNO': 'Ventiquattro ore',
    'NOTTE': 'Periodo buio',
    'MATTINA': 'Inizio della giornata',
    'SERA': 'Fine della giornata',
    'OGGI': 'Questo giorno',
    'IERI': 'Giorno passato',
    'DOMANI': 'Giorno futuro',
    'TEMPO': 'Scorre sempre',
    
    # Emozioni e stati
    'AMORE': 'Sentimento profondo',
    'GIOIA': 'Sentimento di felicità',
    'PAURA': 'Sentimento di timore',
    'RABBIA': 'Sentimento di collera',
    'PACE': 'Assenza di guerra',
    'GUERRA': 'Conflitto armato',
    'VITA': 'Esistenza',
    'MORTE': 'Fine dell\'esistenza',
    'SALUTE': 'Stato di benessere',
    'MALATTIA': 'Stato di malessere',
    
    # Numeri (scritti)
    'UNO': 'Primo numero',
    'DUE': 'Secondo numero',
    'TRE': 'Terzo numero',
    'QUATTRO': 'Quarto numero',
    'CINQUE': 'Quinto numero',
    'DIECI': 'Decimo numero',
    'CENTO': 'Centesimo numero',
    
    # Verbi comuni (infiniti)
    'ESSERE': 'Verbo di esistenza',
    'AVERE': 'Verbo di possesso',
    'FARE': 'Verbo di azione',
    'DIRE': 'Verbo di parola',
    'ANDARE': 'Verbo di movimento',
    'VENIRE': 'Verbo di arrivo',
    'VEDERE': 'Verbo di vista',
    'SENTIRE': 'Verbo di udito',
    'MANGIARE': 'Verbo del nutrimento',
    'BERE': 'Verbo del dissetarsi',
    'DORMIRE': 'Verbo del riposo',
    'CORRERE': 'Verbo della velocità',
    'CAMMINARE': 'Verbo del movimento lento',
    'PARLARE': 'Verbo della comunicazione',
    'CANTARE': 'Verbo della musica',
    'BALLARE': 'Verbo del movimento ritmico',
    'RIDERE': 'Verbo dell\'allegria',
    'PIANGERE': 'Verbo della tristezza',
    'STUDIARE': 'Verbo dell\'apprendimento',
    'LAVORARE': 'Verbo del lavoro',
    
    # Aggettivi
    'GRANDE': 'Di dimensioni ampie',
    'PICCOLO': 'Di dimensioni ridotte',
    'ALTO': 'Di statura elevata',
    'BASSO': 'Di statura ridotta',
    'LUNGO': 'Di lunghezza estesa',
    'CORTO': 'Di lunghezza ridotta',
    'NUOVO': 'Appena fatto',
    'VECCHIO': 'Di età avanzata',
    'GIOVANE': 'Di poca età',
    'BELLO': 'Di aspetto gradevole',
    'BRUTTO': 'Di aspetto sgradevole',
    'BUONO': 'Di qualità positiva',
    'CATTIVO': 'Di qualità negativa',
    'CALDO': 'Di temperatura elevata',
    'FREDDO': 'Di temperatura bassa',
    'VELOCE': 'Di andatura rapida',
    'LENTO': 'Di andatura lenta',
    
    # Mestieri
    'MEDICO': 'Cura i malati',
    'MAESTRO': 'Insegna a scuola',
    'OPERAIO': 'Lavora in fabbrica',
    'CONTADINO': 'Lavora nei campi',
    'CUOCO': 'Prepara i cibi',
    'PITTORE': 'Dipinge quadri',
    'MUSICISTA': 'Suona strumenti',
    'SCRITTORE': 'Scrive libri',
    'POETA': 'Scrive versi',
    'GIORNALISTA': 'Scrive notizie',
    'GATTO': 'Felino domestico che fa le fusa',
    'CANE': 'Il migliore amico dell\'uomo',
    'CASA': 'Dolce dimora dove si abita',
    'SOLE': 'Stella che illumina il giorno',
    'MARE': 'Distesa azzurra e salata',
    'AMORE': 'Sentimento profondo',
}

@dataclass
class WordPlacement:
    word: str
    row: int
    col: int
    direction: str  # 'across' or 'down'
    number: int

class SimpleCrosswordGenerator:
    def __init__(self, width: int = 15, height: int = 15, dictionary_file: str = 'data/dizionario_italiano.json'):
        self.width = width
        self.height = height
        self.grid = [['.' for _ in range(width)] for _ in range(height)]
        self.words = []
        self.placed_words: List[WordPlacement] = []
        self.used_words: Set[str] = set()
        self.clues_dict = load_italian_dictionary(dictionary_file)
        
    def load_words(self, filename: str) -> bool:
        """Load words from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                words = [line.strip().upper() for line in f if line.strip()]
                # Filter valid words (3-15 characters, only letters)
                self.words = [w for w in words if 3 <= len(w) <= 15 and w.isalpha()]
            print(f"Loaded {len(self.words)} words")
            return True
        except FileNotFoundError:
            print(f"Warning: Could not load {filename}, using dictionary words as fallback")
            self.words = list(self.clues_dict.keys())
            return False
    
    def can_place_word(self, word: str, row: int, col: int, direction: str) -> bool:
        """Check if word can be placed at position"""
        if direction == 'across':
            if col + len(word) > self.width:
                return False
            # Check if word fits and doesn't conflict
            for i, letter in enumerate(word):
                current_cell = self.grid[row][col + i]
                if current_cell != '.' and current_cell != letter:
                    return False
            # Check boundaries (no adjacent words)
            if col > 0 and self.grid[row][col - 1] != '.':
                return False
            if col + len(word) < self.width and self.grid[row][col + len(word)] != '.':
                return False
        else:  # down
            if row + len(word) > self.height:
                return False
            for i, letter in enumerate(word):
                current_cell = self.grid[row + i][col]
                if current_cell != '.' and current_cell != letter:
                    return False
            # Check boundaries
            if row > 0 and self.grid[row - 1][col] != '.':
                return False
            if row + len(word) < self.height and self.grid[row + len(word)][col] != '.':
                return False
        
        return True
    
    def place_word(self, word: str, row: int, col: int, direction: str) -> bool:
        """Place word on grid"""
        if not self.can_place_word(word, row, col, direction):
            return False
            
        # Place the word
        if direction == 'across':
            for i, letter in enumerate(word):
                self.grid[row][col + i] = letter
        else:
            for i, letter in enumerate(word):
                self.grid[row + i][col] = letter
        
        # Add to placed words
        word_placement = WordPlacement(word, row, col, direction, len(self.placed_words) + 1)
        self.placed_words.append(word_placement)
        self.used_words.add(word)
        return True
    
    def find_intersections(self, word: str) -> List[Tuple[int, int, str]]:
        """Find possible intersection points for a word"""
        intersections = []
        
        for placed_word in self.placed_words:
            for i, letter1 in enumerate(word):
                for j, letter2 in enumerate(placed_word.word):
                    if letter1 == letter2:
                        # Calculate position for perpendicular placement
                        if placed_word.direction == 'across':
                            # Place new word vertically
                            new_row = placed_word.row - i
                            new_col = placed_word.col + j
                            if new_row >= 0 and new_row + len(word) <= self.height:
                                intersections.append((new_row, new_col, 'down'))
                        else:
                            # Place new word horizontally
                            new_row = placed_word.row + j
                            new_col = placed_word.col - i
                            if new_col >= 0 and new_col + len(word) <= self.width:
                                intersections.append((new_row, new_col, 'across'))
        
        return intersections
    
    def generate_crossword(self, max_words: int = 10) -> bool:
        """Generate a crossword puzzle"""
        if not self.words:
            print("No words available")
            return False
        
        # Prioritize words from our clue dictionary
        priority_words = [w for w in self.clues_dict.keys() if w in self.words]
        other_words = [w for w in self.words if w not in self.clues_dict]
        
        # Combine lists with priority words first
        available_words = priority_words + other_words
        available_words = [w for w in available_words if w not in self.used_words]
        
        # Place first word in center horizontally - try to use a word from our dictionary
        first_word = None
        for word in available_words[:20]:  # Try first 20 words
            start_row = self.height // 2
            start_col = max(0, (self.width - len(word)) // 2)
            
            if self.place_word(word, start_row, start_col, 'across'):
                first_word = word
                print(f"Placed first word: {first_word}")
                break
                
        if not first_word:
            print("Could not place first word")
            return False
        
        # Try to place remaining words
        words_placed = 1
        attempts = 0
        max_attempts = len(available_words) * 10
        
        while words_placed < max_words and attempts < max_attempts:
            attempts += 1
            available_words = [w for w in self.words if w not in self.used_words]
            
            if not available_words:
                break
                
            word = random.choice(available_words)
            intersections = self.find_intersections(word)
            
            if intersections:
                random.shuffle(intersections)
                placed = False
                
                for row, col, direction in intersections:
                    if self.place_word(word, row, col, direction):
                        print(f"Placed word {words_placed + 1}: {word} at ({row}, {col}) {direction}")
                        words_placed += 1
                        placed = True
                        break
                
                if not placed:
                    # Try placing without intersection (rare)
                    for _ in range(10):
                        row = random.randint(0, self.height - 1)
                        col = random.randint(0, self.width - len(word))
                        direction = random.choice(['across', 'down'])
                        
                        if direction == 'down' and row + len(word) > self.height:
                            continue
                            
                        if self.place_word(word, row, col, direction):
                            print(f"Placed isolated word {words_placed + 1}: {word}")
                            words_placed += 1
                            break
        
        print(f"Successfully placed {words_placed} words")
        return words_placed > 1
    
    def print_grid(self):
        """Print the current grid"""
        print("\nCrossword Grid:")
        for row in self.grid:
            print(' '.join('█' if cell == '.' else cell for cell in row))
    
    def get_clue(self, word: str) -> str:
        """Get clue for word"""
        if word in self.clues_dict:
            return self.clues_dict[word]
        
        # Generate simple descriptive clue in Italian
        if len(word) <= 3:
            return f"Parola corta ({len(word)} lettere)"
        elif len(word) <= 6:
            return f"Parola media ({len(word)} lettere)"
        else:
            return f"Parola lunga ({len(word)} lettere)"
    
    def generate_html(self, filename: str) -> bool:
        """Generate HTML crossword"""
        try:
            # Create grid HTML
            grid_html = '<table class="crossword-grid" style="border-collapse: collapse;">\n'
            
            # Number the cells
            cell_numbers = {}
            for i, placed_word in enumerate(self.placed_words, 1):
                key = f"{placed_word.row},{placed_word.col}"
                if key not in cell_numbers:
                    cell_numbers[key] = i
                    placed_word.number = i
            
            for row in range(self.height):
                grid_html += '  <tr>\n'
                for col in range(self.width):
                    cell = self.grid[row][col]
                    if cell == '.':
                        grid_html += '    <td style="background: black; width: 30px; height: 30px;"></td>\n'
                    else:
                        cell_key = f"{row},{col}"
                        number = cell_numbers.get(cell_key, '')
                        number_span = f'<span style="position: absolute; top: 2px; left: 2px; font-size: 10px;">{number}</span>' if number else ''
                        grid_html += f'    <td style="border: 1px solid black; width: 30px; height: 30px; position: relative; background: white;">\n'
                        grid_html += f'      {number_span}\n'
                        grid_html += f'      <input type="text" maxlength="1" style="width: 100%; height: 100%; border: none; text-align: center; font-size: 16px; background: transparent;" data-answer="{cell}">\n'
                        grid_html += f'    </td>\n'
                grid_html += '  </tr>\n'
            grid_html += '</table>\n'
            
            # Create clues HTML
            across_clues = []
            down_clues = []
            
            for placed_word in sorted(self.placed_words, key=lambda x: x.number):
                clue_text = self.get_clue(placed_word.word)
                clue_line = f"{placed_word.number}. {clue_text} ({len(placed_word.word)})"
                
                if placed_word.direction == 'across':
                    across_clues.append(clue_line)
                else:
                    down_clues.append(clue_line)
            
            clues_html = '<div style="margin-top: 20px;">\n'
            if across_clues:
                clues_html += '  <h3>Across</h3>\n  <ol>\n'
                for clue in across_clues:
                    clues_html += f'    <li>{clue}</li>\n'
                clues_html += '  </ol>\n'
            
            if down_clues:
                clues_html += '  <h3>Down</h3>\n  <ol>\n'
                for clue in down_clues:
                    clues_html += f'    <li>{clue}</li>\n'
                clues_html += '  </ol>\n'
            clues_html += '</div>\n'
            
            # Complete HTML
            html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Simple Crossword</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .crossword-grid {{ border-collapse: collapse; margin: 20px auto; }}
        .crossword-grid td {{ border: 1px solid black; width: 30px; height: 30px; position: relative; }}
        input {{ width: 100%; height: 100%; border: none; text-align: center; font-size: 16px; text-transform: uppercase; }}
        input:focus {{ background-color: #e7f3ff; outline: none; }}
        .clues {{ max-width: 600px; margin: 0 auto; }}
        button {{ background: #007bff; color: white; border: none; padding: 10px 20px; margin: 10px; border-radius: 5px; cursor: pointer; }}
        button:hover {{ background: #0056b3; }}
    </style>
</head>
<body>
    <h1 style="text-align: center;">Simple Crossword Puzzle</h1>
    
    <div style="text-align: center;">
        <button onclick="checkAnswers()">Check Answers</button>
        <button onclick="showSolution()">Show Solution</button>
        <button onclick="clearGrid()">Clear Grid</button>
    </div>
    
    {grid_html}
    
    <div class="clues">
        {clues_html}
    </div>
    
    <script>
        function checkAnswers() {{
            let correct = 0;
            let total = 0;
            const inputs = document.querySelectorAll('input[data-answer]');
            
            inputs.forEach(input => {{
                total++;
                const answer = input.dataset.answer;
                if (input.value.toUpperCase() === answer) {{
                    input.style.backgroundColor = '#d4edda';
                    correct++;
                }} else if (input.value) {{
                    input.style.backgroundColor = '#f8d7da';
                }} else {{
                    input.style.backgroundColor = '';
                }}
            }});
            
            alert(`Correct: ${{correct}}/${{total}}`);
        }}
        
        function showSolution() {{
            const inputs = document.querySelectorAll('input[data-answer]');
            inputs.forEach(input => {{
                input.value = input.dataset.answer;
                input.style.backgroundColor = '#d1ecf1';
            }});
        }}
        
        function clearGrid() {{
            const inputs = document.querySelectorAll('input[data-answer]');
            inputs.forEach(input => {{
                input.value = '';
                input.style.backgroundColor = '';
            }});
        }}
        
        // Auto-advance to next cell
        document.querySelectorAll('input[data-answer]').forEach(input => {{
            input.addEventListener('input', function() {{
                if (this.value.length === 1) {{
                    const nextInput = this.parentElement.nextElementSibling?.querySelector('input') ||
                                    this.parentElement.parentElement.nextElementSibling?.querySelector('input');
                    if (nextInput) nextInput.focus();
                }}
            }});
            
            input.addEventListener('keypress', function(e) {{
                if (!/[a-zA-Z]/.test(e.key)) {{
                    e.preventDefault();
                }}
            }});
        }});
    </script>
</body>
</html>'''
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            print(f"HTML crossword saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error generating HTML: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Generate a simple crossword puzzle')
    parser.add_argument('--width', type=int, default=15, help='Grid width')
    parser.add_argument('--height', type=int, default=15, help='Grid height')
    parser.add_argument('--words-file', type=str, default='data/parole.txt', help='Word list file')
    parser.add_argument('--dictionary-file', type=str, default='data/dizionario_italiano.json', help='Italian dictionary JSON file')
    parser.add_argument('--output', type=str, default='simple_crossword.html', help='Output HTML file')
    parser.add_argument('--max-words', type=int, default=10, help='Maximum number of words')
    
    args = parser.parse_args()
    
    # Create crossword generator
    generator = SimpleCrosswordGenerator(args.width, args.height, args.dictionary_file)
    
    # Load words
    generator.load_words(args.words_file)
    
    # Generate crossword
    print("Generating crossword...")
    success = generator.generate_crossword(args.max_words)
    
    if success:
        generator.print_grid()
        generator.generate_html(args.output)
        print(f"\nCrossword generated successfully!")
        print(f"Words placed: {len(generator.placed_words)}")
        print(f"HTML file: {args.output}")
    else:
        print("Failed to generate crossword")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
