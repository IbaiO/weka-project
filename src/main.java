package src;

import java.io.File;
import java.io.FileReader;
import weka.core.Instances;
import weka.core.converters.ArffSaver;

@SuppressWarnings("all")
public class main {
    public static void main(String[] args) {
        ///////////// HASIERAKETAK /////////////
        if (args.length != 2) {
            System.out.println("Erabilera: java -jar weka-project.jar <input karpetaren path-a> <output extentzioa>");
            System.exit(0);
        }
        String inputPath = args[0]; System.out.println("Sarrerako fitxategia: " + inputPath);
        String outputFile = args[1]; System.out.println("Irteerako fitxategia: " + outputFile);
        Instances instances[] = null;

        ///////////// PROCESAMENDUA /////////////
        instances = datuBilketa.getDB().bildu(inputPath, outputFile);
        Instances instancesTrain = instances[0]; instancesTrain.setClassIndex(instancesTrain.numAttributes() - 1);
        Instances instancesDev = instances[1]; instancesDev.setClassIndex(instancesDev.numAttributes() - 1);
        Instances BoWinstances = NonSparseBoW.getNonSparseBoW().transform(instancesTrain, outputFile);
        Instances BoWinstancesDev = NonSparseBoW.getNonSparseBoW().transform(instancesDev, outputFile);
        // bostgarren laborategia
        // SMO algoritmoa
        // Linear regression algoritmoa        

        ///////////// IDAZKETA /////////////
        ArffSaver saver = new ArffSaver();
        saver.setInstances(BoWinstances);
        try {
            saver.setFile(new File(outputFile));
            saver.writeBatch();
            System.out.println("Datuak gorde dira: " + outputFile);
        } catch (Exception e) {
            System.out.println("ERROREA: Ezin izan da " + outputFile + " fitxategia gorde.");
            e.printStackTrace();
        }
    }
}
