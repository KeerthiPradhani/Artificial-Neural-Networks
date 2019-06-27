import jess.JessException;
import jess.Rete;


public class Test {

    public static void main(String[] args) {
        try {
            Rete r = new Rete();
           r.batch("SecurityEval.clp");
        } catch (JessException ex) {
            System.err.println(ex);
        }
    }
}
