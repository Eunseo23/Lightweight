package com.lightweight;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ast.CompilationUnit;
// import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.stmt.ExpressionStmt;
// import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

// import java.io.BufferedReader;
// import java.io.BufferedWriter;
// import java.io.FileReader;
// import java.io.FileWriter;
// import java.io.IOException;
// import java.util.List;

public class JavaParserStatement {

    // public static void main(String[] args) {
    //     // 입력 파일 경로 지정
    //     String inputFilePathp = "C:\\Users\\UOS\\Desktop\\data\\p_dir\\NotificationService.java"; // 첫 번째 파일
    //     String inputFilePathf = "C:\\Users\\UOS\\Desktop\\data\\f_dir\\NotificationService.java"; // 두 번째 파일

    //     // 각 코드에 대해 처리할 output 파일 경로 설정
    //     String outputFilePathp = "ParsedCodep.java";
    //     String outputFilePathf = "ParsedCodef.java";

    //     try {
    //         // 첫 번째 파일에서 주석 제거
    //         String cleanedCodep = RemoveComments.removeCommentsFromFile(inputFilePathp);
    //         // 첫 번째 파일의 코드 처리
    //         processCode(cleanedCodep, outputFilePathp);
    //         System.out.println("Processed code for first file has been saved to: " + outputFilePathp);

    //         // 두 번째 파일에서 주석 제거
    //         String cleanedCodef = RemoveComments.removeCommentsFromFile(inputFilePathf);
    //         // 두 번째 파일의 코드 처리
    //         processCode(cleanedCodef, outputFilePathf);
    //         System.out.println("Processed code for second file has been saved to: " + outputFilePathf);
    //     } catch (IOException e) {
    //         System.out.println("An error occurred: " + e.getMessage());
    //     }
    // }

    public static String processCode(String code) {
        try {
            JavaParser parser = new JavaParser();
            CompilationUnit cu = parser.parse(code).getResult().orElseThrow();

            // Modify the code to ensure each expression statement ends with a semicolon
            cu.accept(new VoidVisitorAdapter<Void>() {
                @Override
                public void visit(ExpressionStmt stmt, Void arg) {
                    super.visit(stmt, arg);
                    // Convert the statement to string and add a semicolon if necessary
                    String statementStr = stmt.toString().trim();
                    if (!statementStr.endsWith(";")) {
                        // Adding a semicolon to the statement
                        stmt.setExpression(stmt.getExpression().toString() + ";");
                    }
                }
            }, null);

            // Return the modified code as a string
            return cu.toString();

        } catch (Exception e) {
            e.printStackTrace();
            return code; // 예외 발생 시 원본 코드 반환
        }
    }
}