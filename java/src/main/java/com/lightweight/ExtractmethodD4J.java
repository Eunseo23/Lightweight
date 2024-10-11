package com.lightweight;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.visitor.GenericVisitorAdapter;
import com.github.javaparser.ast.visitor.Visitable;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.Statement;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Optional;
import java.util.regex.Pattern;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import com.lightweight.JavaParserStatement;

public class ExtractmethodD4J {
    private static String extractedMethodContent;

    public static void main(String[] args) throws IOException {
        String javaFilePath = "E:\\Math_lightweight\\Math_81\\EigenDecompositionImpl.java";
        int targetLineNumber = 905;

        // 파일 내용을 읽어와서 processCode 함수에 적용
        String fileContent = new String(Files.readAllBytes(Paths.get(javaFilePath)));

        // processCode 함수 호출하여 변환된 코드 출력
        String processedCode = JavaParserStatement.processCode(fileContent);

        // JavaParser 객체 생성
        JavaParser parser = new JavaParser();

        // processedCode를 다시 파싱하여 CompilationUnit으로 변환
        CompilationUnit cu = parser.parse(processedCode).getResult().orElseThrow(() ->
            new IllegalArgumentException("Parsing failed"));

        // 주석 제거 메서드 호출
        removeComments(cu);

        // 주석이 제거된 코드 출력
        String processedCodeWithoutComments = cu.toString();
        System.out.println("주석이 제거된 변환된 코드:");
        System.out.println(processedCodeWithoutComments);

        // 주석이 제거된 코드를 파일로 저장
        Files.write(Paths.get("statementfile.txt"), processedCodeWithoutComments.getBytes(StandardCharsets.UTF_8));

        // 1. 메서드 추출 함수 실행
        extractMethod(javaFilePath, targetLineNumber);
        if (extractedMethodContent != null) {
            // 추출된 메서드에서 인덴트를 제거하여 파일에 저장
            String methodWithoutIndent = removeIndentationAndBlankLines(extractedMethodContent);
            Files.write(Paths.get("Original_buggy_method_by_line.txt"), methodWithoutIndent.getBytes(StandardCharsets.UTF_8));
        }
    }

    // 1. 메서드 추출 함수
    public static void extractMethod(String javaFilePath, int targetLineNumber) throws IOException {
        File file = new File(javaFilePath);
        JavaParser parser = new JavaParser();

        ParseResult<CompilationUnit> parseResult = parser.parse(file);

        if (parseResult.isSuccessful() && parseResult.getResult().isPresent()) {
            CompilationUnit cu = parseResult.getResult().get();

            // 주석 제거
            removeComments(cu);

            // 메서드 단위로 탐색하여 해당 줄이 포함된 메서드를 찾음
            cu.accept(new VoidVisitorAdapter<Void>() {
                @Override
                public void visit(MethodDeclaration md, Void arg) {
                    super.visit(md, arg);

                    Optional<Integer> beginLineOpt = md.getBegin().map(p -> p.line);
                    Optional<Integer> endLineOpt = md.getEnd().map(p -> p.line);

                    if (beginLineOpt.isPresent() && endLineOpt.isPresent()) {
                        int beginLine = beginLineOpt.get();
                        int endLine = endLineOpt.get();

                        if (targetLineNumber >= beginLine && targetLineNumber <= endLine) {
                            extractedMethodContent = md.toString();
                            System.out.println("주석 제거된 추출된 메서드:");
                            System.out.println(extractedMethodContent);
                        }
                    }
                }
            }, null);
        }
    }

    // 주석 제거 메서드
    public static void removeComments(CompilationUnit cu) {
        cu.getAllContainedComments().forEach(Comment::remove);
    }

    // 인덴트 및 공백 라인을 제거하는 메서드
    public static String removeIndentationAndBlankLines(String content) {
        // 각 줄의 앞부분에 있는 공백(탭 또는 스페이스)을 제거하고 공백만 있는 줄은 제외
        String[] lines = content.split("\\r?\\n");
        StringBuilder sb = new StringBuilder();
        for (String line : lines) {
            String trimmedLine = line.stripLeading();  // 앞부분 공백 제거
            if (!trimmedLine.isBlank()) {              // 공백 라인 제외
                sb.append(trimmedLine).append(System.lineSeparator());
            }
        }
        return sb.toString();
    }
}
