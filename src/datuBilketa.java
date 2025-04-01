package src;

import weka.core.Instances;
import weka.core.Attribute;
import weka.core.DenseInstance;

import java.io.File;
import java.io.FileWriter;
import java.io.FileReader;
import java.io.BufferedReader;
import java.util.ArrayList;
import java.util.List;

@SuppressWarnings("all")
public class datuBilketa {

    private static datuBilketa nireDB = null;

    public static datuBilketa getDB() {
        if (nireDB == null) {
            nireDB = new datuBilketa();
        }
        return nireDB;
    }

    public Instances[] bildu(String inPath, String outFile) {
        Instances train = bilduTrain (inPath + "/train", outFile);
        Instances dev = bilduTrain (inPath + "/dev", outFile);
        return new Instances[] {train, dev};        
    }

    private Instances bilduTrain(String inPath, String outFile) {
        try {
            // Direktorioko karpetak irakurri
            File dir = new File(inPath);
            if (!dir.isDirectory()) {
                throw new Exception("Emandako path-a ez da direktorio bat.");
            }

            // Karpetak (klasea) lortu
            File[] classFolders = dir.listFiles(File::isDirectory);

            // .arff fitxategirako atributuak sortu
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("file_content", (List<String>) null)); // Fitxategiaren edukia
            List<String> classValues = new ArrayList<>();
            for (File folder : classFolders) {
                classValues.add(folder.getName()); // Karpeten izenak klase bezala
            }
            attributes.add(new Attribute("class", classValues)); // Klase atributua

            // Datu multzoa sortu
            Instances dataTrain = new Instances("Dataset", attributes, 0);
            dataTrain.setClassIndex(1); // "class" atributua klase atributua da

            // Karpetak zeharkatu eta instantziak gehitu
            for (File folder : classFolders) {
                File[] files = folder.listFiles(File::isFile);
                if (files != null) {
                    for (File file : files) {
                        // Fitxategiaren edukia irakurri
                        StringBuilder contentBuilder = new StringBuilder();
                        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                            String line;
                            while ((line = reader.readLine()) != null) {
                                contentBuilder.append(line).append("\n");
                            }
                        } catch (Exception e) {
                            System.out.println("Errorea fitxategia irakurtzean: " + file.getName());
                            e.printStackTrace();
                        }

                        // Instantzia sortu eta datuak gehitu
                        DenseInstance instance = new DenseInstance(2);
                        instance.setValue(attributes.get(0), contentBuilder.toString()); // Fitxategiaren edukia
                        instance.setValue(attributes.get(1), folder.getName()); // Klasea (karpetaren izena)
                        dataTrain.add(instance);
                    }
                }
            }

            // .arff fitxategia gorde
            FileWriter writer = new FileWriter(outFile + "Train.arff");
            writer.write(dataTrain.toString());
            writer.close();

            System.out.println("Fitxategia .arff formatuan ongi sortu da hemen: " + outFile + "Train.arff");

            return dataTrain;
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }

    private Instances bilduDev(String inPath, String outFile) {
        try {
            // Direktorioko karpetak irakurri
            File dir = new File(inPath);
            if (!dir.isDirectory()) {
                throw new Exception("Emandako path-a ez da direktorio bat.");
            }

            // Karpetak (klasea) lortu
            File[] classFolders = dir.listFiles(File::isDirectory);

            // .arff fitxategirako atributuak sortu
            ArrayList<Attribute> attributes = new ArrayList<>();
            attributes.add(new Attribute("file_content", (List<String>) null)); // Fitxategiaren edukia
            List<String> classValues = new ArrayList<>();
            for (File folder : classFolders) {
                classValues.add(folder.getName()); // Karpeten izenak klase bezala
            }
            attributes.add(new Attribute("class", classValues)); // Klase atributua

            // Datu multzoa sortu
            Instances dataDev = new Instances("Dataset", attributes, 0);
            dataDev.setClassIndex(1); // "class" atributua klase atributua da

            // Karpetak zeharkatu eta instantziak gehitu
            for (File folder : classFolders) {
                File[] files = folder.listFiles(File::isFile);
                if (files != null) {
                    for (File file : files) {
                        // Fitxategiaren edukia irakurri
                        StringBuilder contentBuilder = new StringBuilder();
                        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
                            String line;
                            while ((line = reader.readLine()) != null) {
                                contentBuilder.append(line).append("\n");
                            }
                        } catch (Exception e) {
                            System.out.println("Errorea fitxategia irakurtzean: " + file.getName());
                            e.printStackTrace();
                        }

                        // Instantzia sortu eta datuak gehitu
                        DenseInstance instance = new DenseInstance(2);
                        instance.setValue(attributes.get(0), contentBuilder.toString()); // Fitxategiaren edukia
                        instance.setValue(attributes.get(1), folder.getName()); // Klasea (karpetaren izena)
                        dataDev.add(instance);
                    }
                }
            }

            // .arff fitxategia gorde
            FileWriter writer = new FileWriter(outFile + "Dev.arff");
            writer.write(dataDev.toString());
            writer.close();

            System.out.println("Fitxategia .arff formatuan ongi sortu da hemen: " + outFile + "Dev.arff");

            return dataDev;
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
            e.printStackTrace();
            return null;
        }
    }
}