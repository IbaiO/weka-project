package src;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMO;

import weka.core.Instance;
import weka.core.Instances;
import weka.filters.unsupervised.attribute.StringToWordVector;
import weka.filters.Filter;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;


public class iragarri {
    public static void main(Instances trainSet, Instances testSet, String mota) throws Exception {
        if (trainSet == null || testSet == null) {
            System.out.println("Errorea: Ezin izan da datu multzoak kargatu.");
            return;
        }

        // Configurar índice de clase
        trainSet.setClassIndex(0);
        testSet.setClassIndex(0);

        // Crear modelo de regresión lineal
        LinearRegression modelLR = linearRegression.main(trainSet);
        SMO[] modelSMO = sMO.main(trainSet);
        SMO modelSMO1 = modelSMO[0];
        SMO modelSMO2 = modelSMO[1];
        SMO modelSMO3 = modelSMO[2];
        if (modelLR != null) {
            // Preprocesar el conjunto de prueba
            Instances processedTestSet = preprocessTestData(testSet, trainSet);
            if (processedTestSet == null) {
                System.out.println("Errorea: Ezin izan da test datuak aurreprozesatu.");
                return;
            }

         
            // Realizar predicciones
            iragarketakEgin(modelLR, processedTestSet, "lineal", mota);
        }

        if (modelSMO != null) {
            // Preprocesar el conjunto de prueba
            Instances processedTestSet = preprocessTestData(testSet, trainSet);
            if (processedTestSet == null) {
                System.out.println("Errorea: Ezin izan da test datuak aurreprozesatu.");
                return;
            }

            // Realizar predicciones
            iragarketakEgin(modelSMO1, processedTestSet, "SMO1", mota);
            iragarketakEgin(modelSMO2, processedTestSet, "SMO2", mota);
            iragarketakEgin(modelSMO3, processedTestSet, "SMO3", mota);
        }
    }

    private static void iragarketakEgin(Classifier model, Instances RAWinstances, String modelType, String mota) throws Exception {

        if (modelType.equals("lineal")) {
            String outputFilePath = "src/emaitzak/iragarpena_"+mota+"_LinearRegression.txt";
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
                writer.write("Iragarpenak Linear Regrssion erabiliz:\n"+"  - konf: Konfiantza maila (%)\n"+"            %0 (Neg) -===========- %100 (Pos)\n"+"  - k: Klasea (Pos/Neg)\n\n");
                // Escribir predicciones
                for (int i = 0; i < RAWinstances.numInstances(); i++) {
                    Instance instance = RAWinstances.instance(i);
                    double prediction = model.classifyInstance(instance);
                    boolean predictedClass = prediction > 0.5; // true for Pos, false for Neg
    
    
                    // Escribir línea formateada
                    writer.write((i + 1) + ". instantzia:     konf: %" +String.format("%.2f", prediction * 100)+"       k: "+ (predictedClass ? "Pos" : "Neg")+"\n");
                }
                System.out.println("Predictions saved to: " + outputFilePath);
            } catch (IOException e) {
                System.out.println("ERROREA: Ezin izan da iragarpenak fitxategian gorde.");
                e.printStackTrace();
            } catch (Exception e) {
                System.out.println("ERROREA: Ezin izan da iragarpenik egin.");
                e.printStackTrace();
            }        
        } else if (modelType.equals("SMO1")) {
            String outputFilePath = "src/emaitzak/iragarpena_"+mota+"_SMO_PolyKernel.txt";
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
                // Escribir predicciones
                for (int i = 0; i < RAWinstances.numInstances(); i++) {
                    Instance instance = RAWinstances.instance(i);
                    double prediction = model.classifyInstance(instance);
                    boolean predictedClass = prediction > 0.5; // true for Pos, false for Neg
                    // Escribir línea formateada
                    writer.write((i + 1) + ". instantzia: " + (predictedClass ? "Pos" : "Neg")+"\n");
                }
                System.out.println("Predictions saved to: " + outputFilePath);
            } catch (IOException e) {
                System.out.println("ERROREA: Ezin izan da iragarpenak fitxategian gorde.");
                e.printStackTrace();
            } catch (Exception e) {
                System.out.println("ERROREA: Ezin izan da iragarpenik egin.");
                e.printStackTrace();
            }     
        } else if (modelType.equals("SMO2")) {
            String outputFilePath = "src/emaitzak/iragarpena_"+mota+"_SMO_RBFKernel.txt";
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
                // Escribir predicciones
                for (int i = 0; i < RAWinstances.numInstances(); i++) {
                    Instance instance = RAWinstances.instance(i);
                    double prediction = model.classifyInstance(instance);
                    boolean predictedClass = prediction > 0.5; // true for Pos, false for Neg
                    // Escribir línea formateada
                    writer.write((i + 1) + ". instantzia: " + (predictedClass ? "Pos" : "Neg")+"\n");
                }
                System.out.println("Predictions saved to: " + outputFilePath);
            } catch (IOException e) {
                System.out.println("ERROREA: Ezin izan da iragarpenak fitxategian gorde.");
                e.printStackTrace();
            } catch (Exception e) {
                System.out.println("ERROREA: Ezin izan da iragarpenik egin.");
                e.printStackTrace();
            }     
        } else if (modelType.equals("SMO3")) {
            String outputFilePath = "src/emaitzak/iragarpena_"+mota+"_SMO_PukKernel.txt";
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
                // Escribir predicciones
                for (int i = 0; i < RAWinstances.numInstances(); i++) {
                    Instance instance = RAWinstances.instance(i);
                    double prediction = model.classifyInstance(instance);
                    boolean predictedClass = prediction > 0.5; // true for Pos, false for Neg
                    // Escribir línea formateada
                    writer.write((i + 1) + ". instantzia: " + (predictedClass ? "Pos" : "Neg")+"\n");
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
    }

    private static Instances preprocessTestData(Instances testSet, Instances trainSet) {
        try {
            // Aplicar el mismo preprocesamiento que el conjunto de entrenamiento
            StringToWordVector stringToWordVector = new StringToWordVector();
            stringToWordVector.setInputFormat(trainSet); // Usar el formato del conjunto de entrenamiento
            Instances processedTestSet = Filter.useFilter(testSet, stringToWordVector);

            // Manejar valores faltantes
            weka.filters.unsupervised.attribute.ReplaceMissingValues replaceMissingValues = new weka.filters.unsupervised.attribute.ReplaceMissingValues();
            replaceMissingValues.setInputFormat(processedTestSet);
            processedTestSet = Filter.useFilter(processedTestSet, replaceMissingValues);

            // Configurar índice de clase
            processedTestSet.setClassIndex(0);

            return processedTestSet;
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da test datuak aurreprozesatu.");
            e.printStackTrace();
            return null;
        }
    }
}