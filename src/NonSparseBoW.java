package src;

import java.io.File;
import java.io.FileWriter;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.regex.Pattern;

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;

public class NonSparseBoW {
    private static NonSparseBoW nireNonSparseBoW = null;

    public static NonSparseBoW getNonSparseBoW() {
        if (nireNonSparseBoW == null) {
            nireNonSparseBoW = new NonSparseBoW();
        } 
        return nireNonSparseBoW;
    }

    public Instances transformTrain(Instances data) {
        Instances dataGarbi = datu_garbiketa(data); // Datuak garbitu
        Instances BoWData = transformToBoW(dataGarbi); // Bag of Words formatura aldatu
        Instances NonSparseBoWData = transformToBoWNonSparse(BoWData); // Non-Sparse formatura aldatu
        Instances filteredData = filterAttributesByInfoGain(NonSparseBoWData); // InfoGain bidez atributuak iragazi
        Instances rankedData = rankAttributesByInfoGain(filteredData); // Atributuak InfoGain bidez ordenatu
        Instances reorganizedData = reorderAttributesByDictionary(rankedData); // Reorder attributes in TrainBoW based on the updated dictionary (updated by the rank of InfoGain)
        Instances normalizedData = normalizeData(reorganizedData); // Datuak normalizatu
        
        normalizedData.setClassIndex(0); // Klase atributua lehen atributua izan dadila
        normalizedData.randomize(new java.util.Random(81)); // Randomizatu seed 81 erabiliz

        return normalizedData;
    }

    public Instances transformDevTest(Instances data) {
        try {
            Instances datuak = datu_garbiketa(data); // Datuak garbitu
            datuak.setClassIndex(datuak.numAttributes() - 1); // Azken atributua klasea da

            FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();
            filter.setDictionaryFile(new File("datuak/dictionary.txt")); // Kargatu hiztegia fitxategitik
            filter.setLowerCaseTokens(true); // Testua letra xehez bihurtu
            filter.setOutputWordCounts(false); // Hitz kopurua erabili beharrean, presentzia binarioa erabili
            filter.setAttributeIndices("first"); // Aplikatu atributu guztiei
            filter.setInputFormat(datuak);

            Instances filteredData = Filter.useFilter(datuak, filter);
            filteredData.setClassIndex(filteredData.numAttributes() - 1); // Klase atributua ezarri

            Instances NonSparseBoWData = transformToBoWNonSparse(filteredData); // Non-Sparse formatura aldatu
            Instances normalizedData = normalizeData(NonSparseBoWData); // Datuak normalizatu

            normalizedData.setClassIndex(0); // Klase atributua lehen atributua izan dadila
            normalizedData.randomize(new Random(81)); // Randomizatu seed 81 erabiliz

            return normalizedData;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da transformDevTest metodoa burutu.");
            e.printStackTrace();
            return null;
        }
    }

    private Instances datu_garbiketa(Instances datuak) {
        Pattern hashtagPattern = Pattern.compile("#\\w+");
        Pattern punctuationPattern = Pattern.compile("\\p{Punct}");
        Pattern classWordPattern = Pattern.compile("\\b\\w*class\\w*\\b"); // "class" hitza duten hitzak

        for (int i = 0; i < datuak.numInstances(); i++) {
            Instance instance = datuak.instance(i);
            for (int j = 0; j < instance.numAttributes(); j++) {
                if (instance.attribute(j).isString()) {
                    String text = instance.stringValue(j);
                    text = text.replaceAll("\"", ""); // Komatxoak kendu
                    text = text.toLowerCase(); // Letra xehez bihurtu
                    text = hashtagPattern.matcher(text).replaceAll(""); // Hashtag-ak kendu
                    text = punctuationPattern.matcher(text).replaceAll(""); // Puntuazioak kendu
                    text = classWordPattern.matcher(text).replaceAll(""); // "class" hitzak kendu
                    text = text.replaceAll("\\s+", " "); // Espazio gehiegizkoak kendu
                    instance.setValue(j, text);
                }
            }
        }
        return datuak;
    }

    private Instances filterAttributesByInfoGain(Instances datuak) {
        InfoGainAttributeEval evaluator = new InfoGainAttributeEval(); // InfoGain ebaluatzailea
        try {
            // Ebaluatzailea konfiguratu instantziekin
            datuak.setClassIndex(0);
            evaluator.buildEvaluator(datuak);

            // InfoGain balioak lortu atributu bakoitzerako
            double[] infoGainValues = new double[datuak.numAttributes()];
            for (int i = 0; i < datuak.numAttributes(); i++) {
                infoGainValues[i] = evaluator.evaluateAttribute(i);
            }
            
            // Instantzien kopia sortu atributuak iragazteko
            Instances filteredTrain = new Instances(datuak);

            // InfoGain balioa 0 den atributuak ezabatu, klase atributua izan ezik
            int classIndex = datuak.classIndex();
            for (int i = infoGainValues.length - 1; i >= 0; i--) {
                if (i != classIndex && infoGainValues[i] == 0) { 
                    // Atributuak ezabatu infoGain zero bada eta klasea ez bada
                    filteredTrain.deleteAttributeAt(i);
                }
            }
            filteredTrain.setClassIndex(0); // Klase atributua ezarri iragazitako multzoan

            // Atributuen kopurua inprimatu iragazi aurretik eta ondoren
            System.out.println("Atributuen kopurua iragazi aurretik: " + datuak.numAttributes());
            System.out.println("Atributuen kopurua iragazi ondoren: " + filteredTrain.numAttributes());

            // Hiztegia eguneratu
            updateDictionary(filteredTrain);

            return filteredTrain;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da InfoGain bidezko atributuen iragazketa burutu.");
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    private void updateDictionary(Instances filteredTrain) {
        try {
            // dictionary.txt fitxategia irakurri
            File dictionaryFile = new File("datuak/dictionary.txt");
            List<String> dictionaryWords = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(dictionaryFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    // Hiztegiaren hitzak atera (metadatuak baztertu, adibidez "@@@numDocs")
                    if (!line.startsWith("@") && line.contains(",")) {
                        String word = line.split(",")[0].trim();
                        dictionaryWords.add(word);
                    }
                }
            }

            // filteredTrain atributuen izenak lortu
            Set<String> filteredAttributes = new HashSet<>();
            for (int i = 0; i < filteredTrain.numAttributes(); i++) {
                filteredAttributes.add(filteredTrain.attribute(i).name());
            }

            // Hiztegiaren hitzak iragazi filteredTrain-en ez daudenak baztertuz
            List<String> updatedDictionary = new ArrayList<>();
            for (String word : dictionaryWords) {
                if (filteredAttributes.contains(word)) {
                    updatedDictionary.add(word);
                }
            }

            // Hiztegi berria fitxategian idatzi
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(dictionaryFile))) {
                for (String word : updatedDictionary) {
                    writer.write(word + ",1"); // Formatoa egokitu behar izanez gero
                    writer.newLine();
                }
            }

            System.out.println("Hiztegia eguneratu da " + updatedDictionary.size() + " hitzekin.");
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da dictionary.txt eguneratu.");
            e.printStackTrace();
        }
    }

    public Instances transformToBoW(Instances data) {
        data.setClassIndex(data.numAttributes() - 1); // Azken atributua klasea da
        
        StringToWordVector filter = new StringToWordVector();
        filter.setLowerCaseTokens(true); // Letra xehez jarri testua
        filter.setOutputWordCounts(false); // Ez zenbatu hitzak, bakarrik presentzia (binarioa)
        filter.setAttributeIndices("first"); // Atributu guztiei aplikatu
        filter.setDoNotOperateOnPerClassBasis(true); // Ez erabili klase bakoitzeko
        filter.setTokenizer(new weka.core.tokenizers.WordTokenizer()); // Tokenizatzailea
        filter.setDictionaryFileToSaveTo(new File("datuak/dictionary.txt")); // Irakurri iragazkia fitxategitik

        try {
            filter.setInputFormat(data);
            Instances newData = Filter.useFilter(data, filter);
            return newData;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da Bag of Words transformazioa burutu.");
            e.printStackTrace();
            System.exit(1);
            return null;
        }
    }

    private Instances transformToBoWNonSparse(Instances data) {
        data.setClassIndex(data.numAttributes() - 1); // Azken atributua klasea da
        
        try {
            SparseToNonSparse filter = new SparseToNonSparse();
            filter.setInputFormat(data);
            Instances newData = Filter.useFilter(data, filter);
            return newData;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da Non Sparse transformazioa burutu.");
            e.printStackTrace();
            return null;
        }
    }

    private Instances rankAttributesByInfoGain(Instances datuak) {
        try {
            datuak.setClassIndex(0);
            // Configurar el evaluador de InfoGain
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval();

            // Configurar el método de búsqueda Ranker
            weka.attributeSelection.Ranker ranker = new weka.attributeSelection.Ranker();
            ranker.setNumToSelect(-1); // Seleccionar todos los atributos
            ranker.setThreshold(0.0); // No aplicar umbral

            // Configurar AttributeSelection
            AttributeSelection attributeSelection = new AttributeSelection();
            attributeSelection.setEvaluator(evaluator);
            attributeSelection.setSearch(ranker); // Usar Ranker en lugar de BestFirst

            // Aplicar AttributeSelection al conjunto de datos
            attributeSelection.SelectAttributes(datuak);

            // Obtener los índices de los atributos seleccionados
            int[] rankedAttributes = attributeSelection.selectedAttributes();

            // Crear un nuevo conjunto de datos con los atributos seleccionados
            Instances rankedData = attributeSelection.reduceDimensionality(datuak);

            // Actualizar el archivo dictionary.txt con los atributos ordenados
            updateDictionaryWithRankedAttributes(datuak, rankedAttributes);

            return rankedData;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da atributuak ordenatu InfoGain bidez.");
            e.printStackTrace();
            return null;
        }
    }

    private void updateDictionaryWithRankedAttributes(Instances datuak, int[] rankedAttributes) {
        try {
            // Leer el archivo dictionary.txt
            File dictionaryFile = new File("datuak/dictionary.txt");
            List<String> dictionaryWords = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(dictionaryFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    // Extraer solo las palabras del diccionario (ignorar metadatos como "@@@numDocs")
                    if (!line.startsWith("@") && line.contains(",")) {
                        String word = line.split(",")[0].trim();
                        dictionaryWords.add(word);
                    }
                }
            }

            // Reordenar las palabras del diccionario según los índices de atributos ordenados
            List<String> rankedDictionary = new ArrayList<>();
            for (int index : rankedAttributes) {
                String attributeName = datuak.attribute(index).name();
                if (dictionaryWords.contains(attributeName)) {
                    rankedDictionary.add(attributeName);
                }
            }

            // Escribir el nuevo diccionario en el archivo
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(dictionaryFile))) {
                for (String word : rankedDictionary) {
                    writer.write(word + ",1"); // Puedes ajustar el formato según sea necesario
                    writer.newLine();
                }
            }

            System.out.println("Dictionary aldatu da atributuak InfoGain bidez ordenatuta daudelarik.");
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da dictionary.txt eguneratu ordenatutako atributuekin.");
            e.printStackTrace();
        }
    }

    private Instances reorderAttributesByDictionary(Instances data) {
        try {
            // Leer el archivo dictionary.txt
            File dictionaryFile = new File("datuak/dictionary.txt");
            List<String> dictionaryWords = new ArrayList<>();
            try (BufferedReader reader = new BufferedReader(new FileReader(dictionaryFile))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    // Extraer solo las palabras del diccionario (ignorar metadatos como "@@@numDocs")
                    if (!line.startsWith("@") && line.contains(",")) {
                        String word = line.split(",")[0].trim();
                        dictionaryWords.add(word);
                    }
                }
            }

            // Crear una nueva lista de atributos ordenados según el diccionario
            ArrayList<Attribute> reorderedAttributes = new ArrayList<>();
            reorderedAttributes.add(data.classAttribute()); // Añadir el atributo de clase al principio
            for (String word : dictionaryWords) {
                Attribute attribute = data.attribute(word);
                if (attribute != null && !attribute.equals(data.classAttribute())) {
                    reorderedAttributes.add(attribute);
                }
            }

            // Crear un nuevo conjunto de datos con los atributos reordenados
            Instances reorderedData = new Instances(data.relationName(), reorderedAttributes, data.numInstances());
            reorderedData.setClassIndex(0); // Establecer el índice de clase como el primero

            // Copiar las instancias al nuevo conjunto de datos
            for (int i = 0; i < data.numInstances(); i++) {
                Instance instance = data.instance(i);
                DenseInstance newInstance = new DenseInstance(reorderedData.numAttributes());
                newInstance.setDataset(reorderedData);

                for (int j = 0; j < reorderedAttributes.size(); j++) {
                    Attribute attribute = reorderedAttributes.get(j);
                    if (attribute.isNumeric()) {
                        newInstance.setValue(attribute, instance.value(data.attribute(attribute.name())));
                    } else if (attribute.isNominal()) {
                        newInstance.setValue(attribute, instance.stringValue(data.attribute(attribute.name())));
                    }
                }

                reorderedData.add(newInstance);
            }

            return reorderedData;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan dira atributuak berrantolatu dictionary.txt fitxategiaren arabera.");
            e.printStackTrace();
            return data; // Return the original data if reordering fails
        }
    }

    private Instances normalizeData(Instances data) {
        try {
            data.setClassIndex(0);
            weka.filters.unsupervised.attribute.Normalize normalize = new weka.filters.unsupervised.attribute.Normalize();
            normalize.setInputFormat(data);
            return Filter.useFilter(data, normalize);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da datuak normalizatu.");
            e.printStackTrace();
            return data; // Return original data if normalization fails
        }
    }
}
