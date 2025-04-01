package src;

import weka.classifiers.functions.LinearRegression;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.Filter;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class iragarri {
    public static void main(String[] args) throws Exception {
        // Entrenamiento y prueba (inputs manuales)
        String trainSource = "MiPollaBoW.arff";
        String testSource = "MiPolla2BoW.arff";

        // Cargar conjuntos de datos
        Instances trainSet = loadData(trainSource);
        Instances testSet = loadData(testSource);

        if (trainSet == null || testSet == null) {
            System.out.println("Errorea: Ezin izan da datu multzoak kargatu.");
            return;
        }

        // Configurar índice de clase
        trainSet.setClassIndex(trainSet.numAttributes() - 1);
        testSet.setClassIndex(testSet.numAttributes() - 1);

        // Crear modelo de regresión lineal
        LinearRegression modelLR = linearRegression.main(trainSet);
        if (modelLR != null) {
            // Preprocesar el conjunto de prueba
            Instances processedTestSet = preprocessTestData(testSet, trainSet);
            if (processedTestSet == null) {
                System.out.println("Errorea: Ezin izan da test datuak aurreprozesatu.");
                return;
            }

            // Realizar predicciones
            iragarketakEgin(modelLR, processedTestSet);
        }

        // Método de depuración
        //eraikiDev();
        ebaluatuDev(modelLR, "probaData/toyStringExample_dev_RAW.arff");
    }

    private static void iragarketakEgin(LinearRegression model, Instances RAWinstances) throws Exception {

        //Instances BoWinstances = NonSparseBoW.getNonSparseBoW().transformToBoW(RAWinstances);
        //Instances NonSparseinstances = NonSparseBoW.getNonSparseBoW().transformToBoWNonSparse(BoWinstances);
        String outputFilePath = "src/emaitzak/iragarpena_LinearRegression.txt";
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
            // Escribir predicciones
            for (int i = 0; i < RAWinstances.numInstances(); i++) {
                Instance instance = RAWinstances.instance(i);
                double prediction = model.classifyInstance(instance);
                boolean predictedClass = prediction > 0.5; // true for Pos, false for Neg

                // Formatear atributos
                StringBuilder attributes = new StringBuilder();
                for (int j = 0; j < instance.numAttributes(); j++) {
                    if (j != instance.classIndex()) { // Omitir el atributo de clase
                        attributes.append(instance.value(j));
                        if (j < instance.numAttributes() - 1) {
                            attributes.append(", ");
                        }
                    }
                }

                // Escribir línea formateada
                writer.write((i + 1) + ". instantzia: " + (predictedClass ? "Pos" : "Neg") + 
                             ", (" + attributes + ")\n");
            }
            System.out.println("Predictions saved to: " + outputFilePath);
        } catch (IOException e) {
            System.out.println("ERROREA: Ezin izan da iragarpenak fitxategian gorde.");
            e.printStackTrace();
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da iragarpenik egin.");
            e.printStackTrace();
        }
    }

    private static Instances preprocessTestData(Instances testSet, Instances trainSet) {
        try {
            // Aplicar el mismo preprocesamiento que el conjunto de entrenamiento
            StringToWordVector stringToWordVector = new StringToWordVector();
            stringToWordVector.setInputFormat(trainSet); // Usar el formato del conjunto de entrenamiento
            Instances processedTestSet = Filter.useFilter(testSet, stringToWordVector);

            // Configurar índice de clase
            processedTestSet.setClassIndex(processedTestSet.numAttributes() - 1);

            return processedTestSet;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da test datuak aurreprozesatu.");
            e.printStackTrace();
            return null;
        }
    }

    private static void ebaluatuDev(LinearRegression modelLR, String devFilePath) {
        System.out.println("Evaluating dev set: " + devFilePath);
        Instances devSet = loadData(devFilePath);

        if (devSet == null) {
            System.out.println("Errorea: Ezin izan da dev datu multzoa kargatu.");
            return;
        }

        // Ensure class index is set
        if (devSet.classIndex() == -1) {
            devSet.setClassIndex(1);
        }

        // Preprocess the dev set to ensure consistency with the training set
        devSet = preprocessDevData(devSet);
    

        String outputFilePath = "src/emaitzak/iragarpenaDev_LinearRegression.txt";
        int correctPredictions = 0;

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
            writer.write("Predictions for dev set:\n");

            for (int i = 0; i < devSet.numInstances(); i++) {
                Instance instance = devSet.instance(i);

                // Predict the class
                double prediction = modelLR.classifyInstance(instance);
                System.out.println("Prediction: " + prediction);
                boolean predictedClass = prediction > 0.5; // true for Pos, false for Neg

                // Get the true class
                boolean trueClass;
                if (devSet.classAttribute().isNominal()) {
                    String classValue = devSet.classAttribute().value((int) instance.value(devSet.classIndex()));
                    System.out.println("Class value: " + classValue);
                    trueClass = classValue.equals("Pos");
                } else if (devSet.classAttribute().isString()) {
                    trueClass = instance.stringValue(devSet.classIndex()).equals("Pos");
                    System.out.println("True class (String): " + instance.stringValue(devSet.classIndex()));
                } else if (devSet.classAttribute().isNumeric()) {
                    trueClass = instance.value(devSet.classIndex()) > 0.5;
                    System.out.println("True class (Numeric): " + instance.value(devSet.classIndex()));
                } else {
                    throw new IllegalArgumentException("Unsupported class attribute type.");
                }

                // Compare prediction with true class
                if (predictedClass == trueClass) {
                    correctPredictions++;
                }

                // Write prediction to file
                writer.write((i + 1) + ". instantzia: Predicted=" + (predictedClass ? "Pos" : "Neg") + 
                             ", True=" + (trueClass ? "Pos" : "Neg") + "\n");
            }

            // Calculate and write accuracy
            double accuracy = (double) correctPredictions / devSet.numInstances() * 100;
            writer.write("\nAccuracy: " + String.format("%.2f", accuracy) + "%\n");
            System.out.println("Dev set evaluation completed. Results saved to: " + outputFilePath);
        } catch (IOException e) {
            System.out.println("ERROREA: Ezin izan da iragarpenak fitxategian gorde.");
            e.printStackTrace();
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da iragarpenik egin.");
            e.printStackTrace();
        }
    }

    private static Instances preprocessDevData(Instances devSet) {
        try {
            // Apply StringToWordVector filter if there are string attributes
            boolean hasStringAttributes = false;
            System.out.println("Checking for string attributes in dev set...");
            for (int i = 0; i < devSet.numAttributes(); i++) {
                if (devSet.attribute(i).isString()) {
                    System.out.println("String attribute found: " + devSet.attribute(i).name());
                    hasStringAttributes = true;
                }
            }

            if (hasStringAttributes) {
                StringToWordVector stringToWordVector = new StringToWordVector();
                stringToWordVector.setInputFormat(devSet);
                devSet = Filter.useFilter(devSet, stringToWordVector);
                System.out.println("StringToWordVector filter applied to dev set.");
            }

            // Ensure class index is set after preprocessing
            devSet.setClassIndex(0);

            return devSet;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da dev datuak aurreprozesatu.");
            e.printStackTrace();
            return null;
        }
    }

    private static void eraikiDev() {
        String devFolderPath = "dev/";
        String outputArffPath = "src/emaitzak/dev_data.arff";

        java.io.File devFolder = new java.io.File(devFolderPath);

        if (!devFolder.exists() || !devFolder.isDirectory()) {
            System.out.println("ERROREA: 'dev' karpeta ez da existitzen edo ez da direktorio bat.");
            return;
        }

        try {
            // Usar la clase ekorketa para convertir el directorio en un archivo .arff
            ekorketa ekorketaInstance = ekorketa.getEkorketa();
            Instances devInstances = ekorketaInstance.ekorketa(devFolderPath, outputArffPath);

            if (devInstances == null) {
                System.out.println("ERROREA: Ezin izan da 'dev' karpeta .arff bihurtu.");
                return;
            }

            System.out.println("Archivo .arff generado correctamente: " + outputArffPath);

            // Usar la clase NonSparseBoW para transformar las instancias
            NonSparseBoW nonSparseBoW = NonSparseBoW.getNonSparseBoW();
            String nonSparseOutputPath = "src/emaitzak/dev_data_nonSparse.arff";
            Instances transformedInstances = nonSparseBoW.transform(devInstances, nonSparseOutputPath);

            System.out.println("NonSparseBoW instances created and saved to: " + nonSparseOutputPath);

        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da 'dev' karpeta prozesatu.");
            e.printStackTrace();
        }
    }

    private static Instances loadData(String dataSource) {
        try {
            DataSource source = new DataSource(dataSource);
            return source.getDataSet();
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da datu multzoa kargatu.");
            e.printStackTrace();
            return null;
        }
    }
}