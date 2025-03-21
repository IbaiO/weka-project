package src;
import java.io.FileReader;
import weka.core.Instances;

public class datuakKargatu {
    public static void main(String[] args) {
        try {
            FileReader fi = new FileReader(args[1]);
            Instances instances = new Instances(fi);
            fi.close();
            instances.setClassIndex(instances.numAttributes() - 1);
        } catch (Exception e) {
            System.out.println("Errorea: " + e.getMessage());
        }
    }
}
