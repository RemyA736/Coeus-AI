import os
import re
import gc
import json
import csv
import spacy
import textstat
import language_tool_python

from os.path import splitext
from heapq import merge
from collections import Counter
from dateparser.search import search_dates

from extras.preprocessing import *


tool = language_tool_python.LanguageTool('es_MX')

class Documento:
    """
    Clase que contiene todos los datos necesarios para describir y analizar un Documento.
    """
    def __init__(self, nombre_archivo, conjunto, texto_manual):
        # Datos proporcionados originalmente
        self.nombre_archivo = nombre_archivo
        self.conjunto = conjunto
        self.texto_manual = texto_manual

        # Datos obtenidos automáticamente
        self.texto_ocr_default = None
        self.texto_ocr_mejorado = None

        # PLN
        self.personas = []
        self.organizaciones = []
        self.lugares = []
        self.fechas = []

    def get_ruta_archivo(self):
        return os.path.join(self.conjunto, self.nombre_archivo)

    def set_texto_ocr_default(self, texto):
        self.texto_ocr_default = texto

    def set_texto_ocr_mejorado(self, texto):
        self.texto_ocr_mejorado = texto

    def add_persona(self, person):
        self.personas.append(person)

    def add_organizacion(self, organizacion):
        self.organizaciones.append(organizacion)

    def add_lugar(self, lugar):
        self.lugares.append(lugar)

    def add_fecha(self, fecha):
        self.fechas.append(fecha)

def gcloud_text_detection(path):
    import io
    from google.cloud import vision
    """Detecta el texto en la imagen de entrada. (GCloud)"""
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    str_text = " ".join([text.description for text in texts])

    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return str_text

def metricas(docOCR, docProcesado):
    import fastwer
    import pandas as pd
    from os.path import splitext
    from textdistance import Sorensen #dice
    from textdistance import Jaccard
    from textdistance import Hamming
    textOCR= open(docOCR,'r')
    textPro= open(docProcesado,'r')
    docProce=textOCR.read()
    docOCR=textPro.read()

    wer= fastwer.score_sent(textOCR.read(), textPro.read(),char_level=False)
    jaccard = textdistance.Jaccard(external=False)
    jacc= jaccard.similarity(docProce,docOCR)

    sorensen = textdistance.Sorensen(external=False)
    dice=sorensen.similarity(docProce,docOCR)

    hamming = textdistance.Hamming(external=False)
    ha=hamming.similarity(docProce,docOCR)
    filename, extension = splitext(docOCR)

    df = pd.DataFrame(columns = ['img_filename', 'Hamming', 'Jaccard', 'Dice'])
    df = df.append({'img_filename': filename, 'Hamming': ha, 'Jaccard': jacc,'Dice': dice }, ignore_index=True)
    print(df)

def azure_text_detection(path):
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
    from msrest.authentication import CognitiveServicesCredentials
    import os
    import time
    from PIL import Image
    """Detecta el texto en la imagen de entrada. (Azure)"""
    subscription_key = "36912a8b16fd43bd967d6f1abfde5080"
    endpoint = "https://aigaminglatamjes.cognitiveservices.azure.com/"
    computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))
    read_image = open(path, "rb")
    try:
        read_response = computervision_client.read_in_stream(read_image, model_version='2021-04-12', reading_order='basic',  raw=True)
    except:
        print("Formato incorrecto.")
    read_operation_location = read_response.headers["Operation-Location"]
    operation_id = read_operation_location.split("/")[-1]
    while True:
        read_result = computervision_client.get_read_result(operation_id)
        if read_result.status.lower () not in ['notstarted', 'running']:
            break
        time.sleep(10)
    str_text=""
    if read_result.status == OperationStatusCodes.succeeded:
        for text_result in read_result.analyze_result.read_results:
            for line in text_result.lines:
                str_text= str_text + line.text +"\n"
    return str_text

def images_filters(path_image, path_folder_save):
    ##Esta función recibe la dirección de una imagen y el directorio en donde se guardara
    from PIL import Image, ImageEnhance
    im=Image.open(path_image)
    enhancer = ImageEnhance.Brightness(im)
    enhanced_im = enhancer.enhance(2)
    enhancer2 = ImageEnhance.Contrast(enhanced_im)
    enhanced_im = enhancer2.enhance(1.5)
    name=path_image.split('/')[-1]
    filename = splitext(name)[0]
    enhanced_im.save(path_folder_save+"PROCESSING_"+filename+".png")
    print("Imagen guardada.....")

def spellCheck(document_text):
    import language_tool_python
    from os.path import splitext
    tool = language_tool_python.LanguageTool('es')
    filename, extension = splitext(document_text)
    fin = open(document_text, "r")
    textoSC = open(filename+"SC.txt", "w+",encoding="utf-8")
    for line in fin:
        cadena = line.replace('-', ' ')
        cadena = tool.correct(cadena)
        textoSC.write(tool.correct(cadena))
    fin.close()
    textoSC.close()

def tesseract_text_detection(path):
    import pytesseract
    import cv2

    """Detecta el texto en la imagen de entrada. (Tesseract)"""
    image = cv2.imread(path)
    custom_config = r'-l spa --oem 3 --psm 3'
    str_text = pytesseract.image_to_string(image, config=custom_config)

    return str_text

def easyocr_text_detection(path):
    import easyocr
    from torch.cuda import is_available as is_gpu_available

    """Detecta el texto en la imagen de entrada. (EasyOCR)"""
    if is_gpu_available():
        reader = easyocr.Reader(['es'], gpu=True)
    else:
        reader = easyocr.Reader(['es'], gpu=False)

    result = reader.readtext(path, rotation_info=[90,180,270])
    str_text=""
    for i in range(len(result)):
        str_text=str_text+result[i][1]+" "

    return str_text

def retrieve_transcriptions_json(path_to_docs_dir, path_to_json):
    """Crea o abre un JSON para almacenar las transcripciones"""
    filenames = [filename for filename in os.listdir(path_to_docs_dir)]
    # Si el JSON con las transcripciones ya existe
    if os.path.isfile(path_to_json):
        print("El archivo de transcripciones ya existe:", path_to_json)
        f = open(path_to_json)
        json_data = json.load(f)
        f.close()

        return json_data
    # Si el JSON con las transcripciones no existe
    else:
        print("Creando el archivo de transcripciones:", path_to_json)
        f = open("extras/transcription_template.json")
        json_template = json.load(f)
        f.close()

        # Obteniendo el nombre del dataset
        dataset_name = path_to_json.split("/")[-1].split(".")[0]
        json_template['dataset'] = dataset_name

        # Creando el diccionario de documentos
        data = {}

        for filename in filenames:
            filepath = os.path.join(path_to_docs_dir, "/documents")
            filepath = os.path.join(filepath, filename)
            data[filename] = {"filepath": filepath}

        # Agregando el diccionario de datos al JSON
        json_template['data'] = data

        # Creando y guardando el JSON
        f = open(path_to_json, "x")
        json.dump(json_template, f)
        f.close()

        return json_template

def rotate_documents(path_to_docs_dir):
    import cv2
    from extras.deskew import deskew

    # Se crea una subcarpeta con los documentos rotados
    rotated_path = os.path.join(path_to_docs_dir, "Rotated")
    if not os.path.exists(rotated_path):
        os.mkdir(rotated_path)

    for filename in os.listdir(path_to_docs_dir):
        filepath = os.path.join(path_to_docs_dir, filename)
        img = cv2.imread(filepath)
        try:
            img_deskew = deskew(img)
            cv2.imwrite(rotated_path+"/"+filename, img_deskew)
        except Exception as e:
            print("El documento",filename,"no se pudo procesar:", e)

    print("Se completó la corrección de orientación de los documentos.")

def languagetool_spell_check(document_text):
    import language_tool_python
    tool = language_tool_python.LanguageTool('es')

    return tool.correct(document_text)

def json_to_txt_transcriptions(path_to_json, path_to_docs_dir, doc_types=[]):
    if os.path.isfile(path_to_json):
        print("Cargando datos:", path_to_json)
        f = open(path_to_json)
        json_data = json.load(f)
        f.close()

        # Defining paths to save txt documents
        azure_path = os.path.join(path_to_docs_dir, "azure_transcriptions")
        easyocr_path = os.path.join(path_to_docs_dir, "easyocr_transcriptions")
        corrected_path = os.path.join(path_to_docs_dir, "corrected_transcriptions")

        print("Escribiendo transcripciones a txt.")
        for document in json_data['data'].items():
            document_name = document[0]

            if doc_types[0] == None:
                return "No se especificó el tipo de documentos."

            if "azure" in doc_types:
                # Saving Azure transcriptions to txt
                azure_transcription = document[1]['texto_azure']
                final_str = document_name + "\n\n" + azure_transcription
                filename = document_name+".txt"
                azure_f = open(os.path.join(azure_path, filename), "w")
                azure_f.write(final_str)
                azure_f.close()

            if "easyocr" in doc_types:
                # Saving EasyOCR transcriptions to txt
                easy_ocr_transcription = document[1]['texto_easyocr']
                final_str = document_name + "\n\n" + easy_ocr_transcription
                filename = document_name+".txt"
                easyocr_f = open(os.path.join(easyocr_path, filename), "w")
                easyocr_f.write(final_str)
                easyocr_f.close()

            if "corrected" in doc_types:
                # Saving corrected transcriptions to txt
                azure_transcription = languagetool_spell_check(document[1]['texto_azure'])
                final_str = document_name + "\n\n" + azure_transcription
                filename = document_name+".txt"
                easyocr_f = open(os.path.join(corrected_path, filename), "w")
                easyocr_f.write(final_str)
                easyocr_f.close()

        print("Documentos TXT creados con éxito.")
    else:
        print("ERROR: El archivo JSON especificado no existe.")

def metricasSRF(docOCR, docProcesado):
    from textdistance import Sorensen #dice
    from textdistance import Jaccard
    from textdistance import Hamming
    from textdistance import Cosine

    jaccard = Jaccard(external=False)
    jacc= jaccard.similarity(docProcesado,docOCR)

    sorensen = Sorensen(external=False)
    dice=sorensen.similarity(docProcesado,docOCR)

    hamming = Hamming(external=False)
    ha=hamming.similarity(docProcesado,docOCR)

    cosine = Cosine(external=False)
    cos_s=cosine.similarity(docProcesado,docOCR)

    diccionario = {'Hamming' : ha, 'Jaccard' : jacc, 'Dice': dice, 'Cosine': cos_s}

    return diccionario

def evaluate_text_languagetool(texto):

    matches = tool.check(texto)

    return len(matches)

def evaluate_textstat(texto):
    textstat.set_lang('es')

    # Índice de perspicuidad de Szigriszt-Pazos
    szigriszt_pazos = textstat.szigriszt_pazos(texto)

    # Fórmula de comprensibilidad de Gutiérrez de Polini
    gutierrez_polini = textstat.gutierrez_polini(texto)

    # Conteo de monosílabos
    monosilabas = textstat.monosyllabcount(texto)

    return {'szigriszt_pazos': szigriszt_pazos,
           'gutierrez_polini': gutierrez_polini,
           'monosilabas': monosilabas}

def compare_transcriptions_unsupervised(texto1, texto2):
    # Contadores
    t1, t2 = 0, 0

    if texto1 == "" and texto2 != "":
        return texto2
    elif texto1 != "" and texto2 == "":
        return texto1

    # Evaluando con textstat
    tstat1 = evaluate_textstat(texto1)
    tstat2 = evaluate_textstat(texto2)

    # Índice de perspicuidad de Szigriszt-Pazos
    if tstat1['szigriszt_pazos'] > tstat2['szigriszt_pazos']:
        t1 += 1
    elif tstat2['szigriszt_pazos'] > tstat1['szigriszt_pazos']:
        t2 += 1

    # Fórmula de comprensibilidad de Gutiérrez de Polini
    if tstat1['gutierrez_polini'] > tstat2['gutierrez_polini']:
        t1 += 1
    elif tstat2['gutierrez_polini'] > tstat1['gutierrez_polini']:
        t2 += 1

    # Conteo de monosílabos
    if tstat1['monosilabas'] < tstat2['monosilabas']:
        t1 += 1
    elif tstat2['monosilabas'] < tstat1['monosilabas']:
        t2 += 1

    # Conteo de errores detectados por language tool
    if evaluate_text_languagetool(texto1) < evaluate_text_languagetool(texto2):
        t1 += 1
    elif evaluate_text_languagetool(texto2) < evaluate_text_languagetool(texto1):
        t2 += 1

    if t1 > t2:
        return texto1
    else:
        return texto2

def entity_azure(texto,key,endpoint):
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.core.credentials import AzureKeyCredential

    entidades=[]
    credential = AzureKeyCredential(key)
    client = TextAnalyticsClient(endpoint=endpoint, credential=credential)
    try:
        documents = [texto]
        result = client.recognize_entities(documents = documents)[0]
        for entity in result.entities:
            if entity.category=="Person":
                categoria="Persona"
            elif entity.category== "PersonType":
                categoria="Tipo de persona"
            elif entity.category== "Location":
                categoria="Localización"
            elif entity.category== "Organization":
                categoria="Organización"
            elif entity.category== "Event":
                categoria="Evento"
            elif entity.category== "Product":
                categoria="Producto"
            elif entity.category== "Skill":
                categoria="Habilidad"
            elif entity.category== "Address":
                categoria="Dirección"
            elif entity.category== "PhoneNumber":
                categoria="Numero de teléfono"
            elif entity.category== "Email":
                categoria="Correo electronico"
            elif entity.category== "URL":
                categoria="URL"
            elif entity.category== "DateTime":
                categoria="Fecha"
            elif entity.category== "Quantity":
                categoria="Cantidad"
            elif entity.category=="IPAddress":
                categoria="Dirección IP"

            entidades.append([entity.text, categoria])
    except Exception as err:
        print("Encountered exception. {}".format(err))
    return entidades


#Requiered download the model "python -m spacy download model_name"
#Spacy models 1)"es_core_news_sm" 2)"es_core_news_md" 3)"es_core_news_lg"
def get_entities(text, model):
    nlp = model
    doc = nlp(text)

    #Encuentra las entidades en el texto, crea una lista con cada caracter de la entidad y el tipo de entidad
    entidades = []
    for sent in reversed(doc.ents):
        entidad = list(str(nlp(sent.text)))
        entidad.append(str(sent.label_))
        entidades.append(entidad)

    #Convierta las entidades a tuplas ("Entidad","tipo")
    tipo_entidad = ['Localización','Organización','Persona','Producto']
    tipos = ['LOC','ORG','PER','MISC']
    entidad = []
    for item in entidades:
        entity = []
        entity.append('' .join(item[:-1]))
        tipo_ent = tipo_entidad[tipos.index(item[-1])]
        entity.append(tipo_ent)
        entidad.append(entity)

    #Lista de fechas encontradas con search_dates
    listDates=search_dates(text,languages=['es'])
    if listDates:
        for item in listDates:
            entity = []
            if item[0] != 'a':
                entity.append(item[0])
                entity.append('Fecha')
                entidad.append(entity)
    else:
        pass

    return entidad

def extract_expediente(texto):
    # Se extraen todos los expedientes presentes en el documento
    exp = re.compile('[bBjJeExXpPzZ ][eE xX pP zZ][eE xX pP zZ][-. :].{1,12}')
    expedientes = exp.findall(texto)

    filtered = []

    # Ligero filtrado de los expedientes
    for i, element in enumerate(expedientes):
        element = element.replace(" ", "")
        element = element.replace("'", "")
        element = element.replace("~", "-")
        filtered.append(element)

    # Se obtiene el expediente más común
    c = Counter(filtered)
    mc = c.most_common(1)

    if len(mc) > 0:
        return mc[0][0]
    else:
        return "No detectado"

def add_entities_json(path_to_json, path_to_save, credentials):
    f = open(path_to_json)
    predictions_dict = json.load(f)
    f.close()

    # Cargando el modelo de spacy
    ner_model = spacy.load("es_core_news_md") #Carga el modelo

    for i, document in enumerate(predictions_dict.items()):
        texto_azure = predictions_dict[document[0]]['texto_azure_pp_adaptive']
        texto_easy = predictions_dict[document[0]]['texto_easyocr_np']

        if 'entidades' not in predictions_dict[document[0]]:
            #Get the best text between Azure and Easy-OCR
            text = compare_transcriptions_unsupervised(texto_azure, texto_easy)

            if text != "":
                #Get entities from Azure and Spacy
                entities_azure = entity_azure(text, credentials['key'], credentials['endpoint'])
                entities_spacy = get_entities(text, ner_model)

                #Combine the entities
                entidades = list(merge(entities_azure, entities_spacy))

                # Extrayendo el expediente
                expediente = extract_expediente(text)
            else:
                expediente = ''
                entidades = []

            # Se añaden las entidades al diccionario
            predictions_dict[document[0]]['entidades'] = entidades

            # Se añade el número de expediente
            predictions_dict[document[0]]['expediente'] = expediente

            print("Entidades del documento",i+1,":",document[0],"han sido procesadas.")
        else:
            print("El documento",document[0],"ya contiene entidades.")

        with open(path_to_save, 'w') as file:
                json.dump(predictions_dict, file, indent=4)
                file.close()

def inference(path_to_docs, path_to_procs, path_to_predictions):
    if os.path.isfile(path_to_predictions):
        print("El archivo de transcripciones ya existe:", path_to_predictions)
        f = open(path_to_predictions)
        predictions_dict = json.load(f)
        f.close()
    else:
        predictions_dict = {}
        # Primer paso, para crear la estructura
        for i, filename in enumerate(os.listdir(path_to_docs)):
            predictions_dict[filename] = {'expediente': "",
                                          'texto_easyocr_np': "",
                                          'texto_azure_pp_adaptive': ""}

        # Guardando avances
        f = open(path_to_predictions, "w")
        json.dump(predictions_dict, f, indent=4)
        f.close()
        print("El archivo de transcripciones se ha creado con éxito:", path_to_predictions)

    # Extracción de texto con EasyOCR
    for i, filename in enumerate(os.listdir(path_to_docs)):
        texto_easyocr_np = ''

        # Ruta al archivo
        np_path = os.path.join(path_to_docs, filename)

        # Si el OCR de EasyOCR aún no ha sido extraído
        if predictions_dict[filename]['texto_easyocr_np'] == "":
            try:
                texto_easyocr_np = easyocr_text_detection(np_path) # sin pre-procesamiento
            except:
                print("La extracción de texto (EasyOCR) falló para el documento", filename)

        # Añadiendo las predicciones al diccionario
        predictions_dict[filename]['texto_easyocr_np'] = texto_easyocr_np
        print("EasyOCR de documento", filename, "finalizado con éxito.")

        # Guardando avances
        f = open(path_to_predictions, "w")
        json.dump(predictions_dict, f, indent=4)
        f.close()

    # Extracción de texto con Azure
    for i, filename in enumerate(os.listdir(path_to_procs)):
        texto_azure_pp_adaptive = ''

        # Ruta al archivo
        pp_adaptive_path = os.path.join(path_to_procs, filename)

        # Si el OCR de Azure aún no ha sido extraído
        if predictions_dict[filename]['texto_azure_pp_adaptive'] == "":
            try:
                texto_azure_pp_adaptive = azure_text_detection(pp_adaptive_path) # binarización adaptativa
            except:
                print("La extracción de texto (Azure) falló para el documento", filename)

        # Añadiendo las predicciones al diccionario
        predictions_dict[filename]['texto_azure_pp_adaptive'] = texto_azure_pp_adaptive
        print("Azure OCR de documento", filename, "finalizado con éxito.")

        # Guardando avances
        f = open(path_to_predictions, "w")
        json.dump(predictions_dict, f, indent=4)
        f.close()

def json_to_csv_2a(path_to_json, path_to_2a):
    # Se carga el archivo JSON
    f = open(path_to_json)
    predictions_dict = json.load(f)
    f.close()

    with open(path_to_2a, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # Escribiendo el encabezado
        header = ['filename', 'text']
        writer.writerow(header)

        # Iterando sobre los documentos en el JSON
        for i, document in enumerate(predictions_dict.items()):
            # Extrayendo el nombre del documento
            filename = document[0]

            # Seleccionando el mejor texto con la métrica compuesta ponderada
            text = compare_transcriptions_unsupervised(document[1]['texto_easyocr_np'], document[1]['texto_azure_pp_adaptive'])

            # Extrayendo el expediente
            expediente = document[1]['expediente']

            # Preparando la cadena para el CSV
            str_f = expediente+': '+text

            # Escribiendo los datos al CSV
            row = [filename, str_f]
            writer.writerow(row)

            print("2A: Documento", document[0], 'procesado.')

def json_to_csv_2b(path_to_json, path_to_2b):
    # Se carga el archivo JSON
    f = open(path_to_json)
    predictions_dict = json.load(f)
    f.close()

    with open(path_to_2b, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)

        # Escribiendo el encabezado
        header = ['filename', 'label', 'class']
        writer.writerow(header)

        # Iterando sobre los documentos en el JSON
        for i, document in enumerate(predictions_dict.items()):
            # Extrayendo el nombre del documento
            filename = document[0]

            # Obteniendo las entidades
            entidades = document[1]['entidades']

            # Por cada entidad en el documento
            for entity in entidades:
                label = entity[0]
                clase = entity[1]

                if clase == 'Cantidad':
                    continue

                # Escribiendo los datos al CSV
                row = [filename, label, clase]
                writer.writerow(row)

            print("2B: Documento", document[0], 'procesado.')
