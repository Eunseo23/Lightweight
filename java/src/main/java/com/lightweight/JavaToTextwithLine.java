package com.lightweight;
//insertion과 deletion이 발생할 경우 동일한 위치에 빈 줄 넣어준다.
//이후 java파일을 txt파일로 바꿔서 저장한다.

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import com.github.difflib.DiffUtils;
import com.github.difflib.patch.AbstractDelta;
import com.github.difflib.patch.Patch;

public class JavaToTextwithLine {

    public static void main(String[] args) {
        // 파일 경로 설정
        String inputFilePathp = "C:\\Users\\UOS\\Desktop\\data\\p_dir\\NotificationService.java"; // 첫 번째 파일
        String inputFilePathf = "C:\\Users\\UOS\\Desktop\\data\\f_dir\\NotificationService.java"; // 두 번째 파일
        String finalOriginalTextFilePath = "OriginalMethods.txt";
        String finalRevisedTextFilePath = "RevisedMethods.txt";

        try {
            // 1. ExtractBuggyMethod를 사용하여 파일 처리
            List<List<String>> modifiedMethods = ExtractBuggyMethod.extractModifiedMethodsFromFiles(inputFilePathp, inputFilePathf);
            List<String> modifiedOriginalMethods = modifiedMethods.get(0);
            List<String> modifiedRevisedMethods = modifiedMethods.get(1);

            // 빈 줄이 삽입된 결과를 저장할 리스트
            List<String> originalWithBlanks = new ArrayList<>(modifiedOriginalMethods);
            List<String> revisedWithBlanks = new ArrayList<>(modifiedRevisedMethods);

            // 차이점 계산 (원본과 수정본의 메서드에서 직접 비교)
            Patch<String> patch = DiffUtils.diff(originalWithBlanks, revisedWithBlanks);

            // 차이점 적용
            for (AbstractDelta<String> delta : patch.getDeltas()) {
                int originalPosition = delta.getSource().getPosition();
                int revisedPosition = delta.getTarget().getPosition();

                switch (delta.getType()) {
                    case DELETE:
                        // 삭제된 경우 revised에 빈 줄 삽입
                        revisedWithBlanks.add(revisedPosition, "");
                        break;
                    case INSERT:
                        // 추가된 경우 original에 빈 줄 삽입
                        originalWithBlanks.add(originalPosition, "");
                        break;
                    case CHANGE:
                    case EQUAL:
                    default:
                        break;
                }
            }

            // 2. 최종 텍스트 파일로 변환 및 저장

            // OriginalWithBlanks의 내용을 텍스트로 변환
            List<String> originalStrippedLines = originalWithBlanks.stream()
                .map(String::trim) // 앞뒤 공백 제거
                .collect(Collectors.toList());

            // 결과를 텍스트 파일로 저장
            Files.write(Paths.get(finalOriginalTextFilePath), originalStrippedLines);

            // RevisedWithBlanks의 내용을 텍스트로 변환
            List<String> revisedStrippedLines = revisedWithBlanks.stream()
                .map(String::trim) // 앞뒤 공백 제거
                .collect(Collectors.toList());

            // 결과를 텍스트 파일로 저장
            Files.write(Paths.get(finalRevisedTextFilePath), revisedStrippedLines);

            // 결과 출력
            System.out.println("Difference applied and Java files converted to text files successfully.");
            System.out.println("Saved as " + finalOriginalTextFilePath + " and " + finalRevisedTextFilePath);

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}