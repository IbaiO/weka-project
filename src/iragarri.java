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
        String trainSource = "probaData/toyStringExample_train_RAW.arff";
        String testSource = "probaData/toyStringExample_test_RAW.arff";

        // Cargar conjuntos de datos
        Instances trainSet = loadData(trainSource);
        Instances testSet = loadData(testSource);

        if (trainSet == null || testSet == null) {
            System.out.println("Errorea: Ezin izan da datu multzoak kargatu.");
            return;
        }

        // Configurar índice de clase
        if (trainSet.classIndex() == -1) {
            trainSet.setClassIndex(trainSet.numAttributes() - 1);
        }
        if (testSet.classIndex() == -1) {
            testSet.setClassIndex(testSet.numAttributes() - 1);
        }

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
        konprobatuDev();
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
                String predictedClass = prediction > 0.5 ? "Pos" : "Neg";

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
                writer.write((i + 1) + ". instantzia: " + predictedClass + ", (" + attributes + ")\n");
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

    private static void konprobatuDev() {
        System.out.println("Konprobatu dev");
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