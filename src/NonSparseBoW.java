package src;

import java.util.regex.Pattern;
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

import weka.attributeSelection.AttributeSelection;
import weka.attributeSelection.BestFirst;
import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.FixedDictionaryStringToWordVector;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.unsupervised.instance.SparseToNonSparse;


@SuppressWarnings("all")
public class NonSparseBoW {
    private static NonSparseBoW nireNonSparseBoW = null;

    public static NonSparseBoW getNonSparseBoW() {
        if (nireNonSparseBoW == null) {
            nireNonSparseBoW = new NonSparseBoW();
        } return nireNonSparseBoW;
    }

    public Instances transformTrain(Instances data) {
        Instances dataGarbi = datu_garbiketa(data);
        Instances BoWData = transformToBoW(data);
        Instances NonSparseBoWData = transformToBoWNonSparse(BoWData);
        Instances filteredData = filterAttributesByInfoGain(NonSparseBoWData);
        Instances rankedData = rankAttributesByInfoGain(filteredData);        
        rankedData.randomize(new java.util.Random(81)); //Randomized by seed 81
        return rankedData;
    }

    public Instances transformDevTest(Instances data) {
        try {
            // Preprocesar los datos
            Instances datuak = datu_garbiketa(data);
            datuak.setClassIndex(datuak.numAttributes() - 1); // Azken atributua klasea da

            FixedDictionaryStringToWordVector filter = new FixedDictionaryStringToWordVector();
            filter.setDictionaryFile(new File("datuak/dictionary.txt")); // Cargar el diccionario desde el archivo
            filter.setLowerCaseTokens(true); // Convertir texto a minúsculas
            filter.setOutputWordCounts(false); // Usar presencia binaria en lugar de conteo de palabras
            filter.setAttributeIndices("first"); // Aplicar a todos los atributos

            try {
                // Configurar el formato de entrada del filtro
                filter.setInputFormat(datuak);

                // Aplicar el filtro a las instancias
                Instances filteredData = Filter.useFilter(datuak, filter);
                filteredData.setClassIndex(filteredData.numAttributes() - 1); // Set the class index

                // Continuar con el procesamiento
                Instances NonSparseBoWData = transformToBoWNonSparse(filteredData);
                NonSparseBoWData.setClassIndex(0); // Set the class index 
                NonSparseBoWData.randomize(new Random(81)); //Randomized by seed 81
                return NonSparseBoWData;
            } catch (Exception e) {
                System.out.println("ERROREA: Ezin izan da FixedDictionaryStringToWordVector iragazkia burutu.");
                e.printStackTrace();
                return null;
            }
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da transformDevTest metodoa burutu.");
            e.printStackTrace();
            return null;
        }
    }

    private Instances datu_garbiketa(Instances datuak) {
        Pattern hashtagPattern = Pattern.compile("#\\w+");
        Pattern punctuationPattern = Pattern.compile("\\p{Punct}");
        Pattern classWordPattern = Pattern.compile("\\b\\w*class\\w*\\b"); // Palabras que contienen "class"

        for (int i = 0; i < datuak.numInstances(); i++) {
            Instance instance = datuak.instance(i);
            for (int j = 0; j < instance.numAttributes(); j++) {
                if (instance.attribute(j).isString()) {
                    String text = instance.stringValue(j);
                    text = text.replaceAll("\"", ""); // Komatxoak kendu
                    text = text.toLowerCase(); // Letra xehetan bihurtu
                    text = hashtagPattern.matcher(text).replaceAll(""); // Hashtagak kendu
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
        InfoGainAttributeEval evaluator = new InfoGainAttributeEval(); // Evaluador de InfoGain
        try {
            // Configurar el evaluador con las instancias
            datuak.setClassIndex(0);
            evaluator.buildEvaluator(datuak);

            // Obtener los valores de InfoGain para cada atributo
            double[] infoGainValues = new double[datuak.numAttributes()];
            for (int i = 0; i < datuak.numAttributes(); i++) {
                infoGainValues[i] = evaluator.evaluateAttribute(i);
            }
            
            // Crear una copia de las instancias para filtrar los atributos
            Instances filteredTrain = new Instances(datuak);

            // Eliminar los atributos con InfoGain igual a 0, excepto el atributo de clase
            int classIndex = datuak.classIndex();
            for (int i = infoGainValues.length - 1; i >= 0; i--) {
                if (i != classIndex && infoGainValues[i] == 0) { // Ignorar el atributo de clase
                    filteredTrain.deleteAttributeAt(i);
                }
            }
            filteredTrain.setClassIndex(0); // Establecer el índice de clase en el conjunto filtrado

            // Imprimir el número de atributos antes y después de filtrar
            System.out.println("Number of attributes before filtering: " + datuak.numAttributes());
            System.out.println("Number of attributes after filtering: " + filteredTrain.numAttributes());

            // Actualizar dictionary file
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

            // Obtener los nombres de los atributos de filteredTrain
            Set<String> filteredAttributes = new HashSet<>();
            for (int i = 0; i < filteredTrain.numAttributes(); i++) {
                filteredAttributes.add(filteredTrain.attribute(i).name());
            }

            // Filtrar las palabras del diccionario que no están en filteredTrain
            List<String> updatedDictionary = new ArrayList<>();
            for (String word : dictionaryWords) {
                if (filteredAttributes.contains(word)) {
                    updatedDictionary.add(word);
                }
            }

            // Escribir el nuevo diccionario en el archivo
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(dictionaryFile))) {
                for (String word : updatedDictionary) {
                    writer.write(word + ",1"); // Puedes ajustar el formato según sea necesario
                    writer.newLine();
                }
            }

            System.out.println("Dictionary actualizado con " + updatedDictionary.size() + " palabras.");
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
        SparseToNonSparse filter = new SparseToNonSparse();
        try {
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
            // Configurar el evaluador de InfoGain
            InfoGainAttributeEval evaluator = new InfoGainAttributeEval();

            // Configurar el método de búsqueda BestFirst
            BestFirst search = new BestFirst();
            search.setSearchTermination(5); // Configurar el número de terminaciones sin mejora

            // Configurar AttributeSelection
            AttributeSelection attributeSelection = new AttributeSelection();
            attributeSelection.setEvaluator(evaluator);
            attributeSelection.setSearch(search);

            // Aplicar AttributeSelection al conjunto de datos
            attributeSelection.SelectAttributes(datuak);

            // Obtener los índices de los atributos seleccionados
            int[] rankedAttributes = attributeSelection.selectedAttributes();

            // Imprimir los atributos ordenados por importancia
            System.out.println("Atributos ordenados por InfoGain:");
            for (int i = 0; i < rankedAttributes.length; i++) {
                System.out.println((i + 1) + ". " + datuak.attribute(rankedAttributes[i]).name());
            }

            // Crear un nuevo conjunto de datos con los atributos seleccionados
            Instances rankedData = attributeSelection.reduceDimensionality(datuak);

            return rankedData;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da atributuak ordenatu InfoGain bidez.");
            e.printStackTrace();
            return null;
        }
    }
}
