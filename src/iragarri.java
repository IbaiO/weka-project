package src;

import weka.classifiers.Classifier;
import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.SMO;
import weka.core.Instance;
import weka.core.Instances;

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
            // Realizar predicciones
            iragarketakEgin(modelLR, testSet, "lineal", mota);
        }

        if (modelSMO != null) {
            // Realizar predicciones
            iragarketakEgin(modelSMO1, testSet, "SMO1", mota);
            iragarketakEgin(modelSMO2, testSet, "SMO2", mota);
            iragarketakEgin(modelSMO3, testSet, "SMO3", mota);
        }
    }

    private static void iragarketakEgin(Classifier model, Instances instantzia, String modelType, String mota) throws Exception {
        instantzia.setClassIndex(0); // Klasea atributua ezarri instantzie
        if (modelType.equals("lineal")) {
            String outputFilePath = "src/emaitzak/iragarpena_"+mota+"_LinearRegression.txt";
            try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath))) {
                writer.write("Iragarpenak Linear Regrssion erabiliz:\n"+"  - konf: Konfiantza maila (%)\n"+"            %0 (Neg) -===========- %100 (Pos)\n"+"  - k: Klasea (Pos/Neg)\n\n");
                // Escribir predicciones
                for (int i = 0; i < instantzia.numInstances(); i++) {
                    Instance instance = instantzia.instance(i);
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
                for (int i = 0; i < instantzia.numInstances(); i++) {
                    Instance instance = instantzia.instance(i);
                
                    // Obtener la predicción y la distribución de probabilidades
                    double prediction = model.classifyInstance(instance);
                    double[] distribution = model.distributionForInstance(instance);
                    double confidence = distribution[1]; // Probabilidad de la clase "Pos"
                
                    // Determinar la clase predicha
                    String predictedClass = instance.classAttribute().value((int) prediction);
                
                    // Escribir la predicción en el archivo
                    writer.write((i + 1) + ". instantzia: konf: %" + String.format("%.2f", confidence * 100) + " k: " + predictedClass + "\n");
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
                for (int i = 0; i < instantzia.numInstances(); i++) {
                    Instance instance = instantzia.instance(i);
                
                    // Obtener la predicción y la distribución de probabilidades
                    double prediction = model.classifyInstance(instance);
                    double[] distribution = model.distributionForInstance(instance);
                    double confidence = distribution[1]; // Probabilidad de la clase "Pos"
                
                    // Determinar la clase predicha
                    String predictedClass = instance.classAttribute().value((int) prediction);
                
                    // Escribir la predicción en el archivo
                    writer.write((i + 1) + ". instantzia: konf: %" + String.format("%.2f", confidence * 100) + " k: " + predictedClass + "\n");
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
                for (int i = 0; i < instantzia.numInstances(); i++) {
                    Instance instance = instantzia.instance(i);
                
                    // Obtener la predicción y la distribución de probabilidades
                    double prediction = model.classifyInstance(instance);
                    double[] distribution = model.distributionForInstance(instance);
                    double confidence = distribution[1]; // Probabilidad de la clase "Pos"
                
                    // Determinar la clase predicha
                    String predictedClass = instance.classAttribute().value((int) prediction);
                
                    // Escribir la predicción en el archivo
                    writer.write((i + 1) + ". instantzia: konf: %" + String.format("%.2f", confidence * 100) + " k: " + predictedClass + "\n");
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
}