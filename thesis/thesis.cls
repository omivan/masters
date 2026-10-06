% ******************** Клас для написання пояснювальної записки ***********************
% ******************** до атестаційної роботи ступеня бакалавра ***********************
% Зміни Таврова Д. Ю. та Цибульника А. В. в шаблоні, створеному Яковлєвим С. В. (ММЗІ)
% на основі шаблону, створеного Тавровим Д. Ю., Борисенком П. Б. та Дутчаком І. О. (ПМА)

% ------------------------------------------------------------- %
%  1. Ідентифікуємо клас
% ------------------------------------------------------------- %

\ProvidesClass{bachelor_thesis}[2025/05/01 v0.8 -- Пояснювальна записка до бакалаврської кваліфікаційної роботи]
\NeedsTeXFormat{LaTeX2e} % потрібний формат

% ------------------------------------------------------------- %
%  2. Завантажуємо "батьківський клас"
% ------------------------------------------------------------- %

\LoadClass[a4paper,14pt,ukrainian,oneside]{extreport}

% ------------------------------------------------------------- %
%  3. Завантажуємо базові пакети
% ------------------------------------------------------------- %

% мовні пакети
\RequirePackage{cmap}                                   % для кодування шрифтів у pdf
%\RequirePackage[T2A]{fontenc}                          % пакет підтримки кирилічних шрифтів pdflatex
%\RequirePackage[utf8]{inputenc}                        % пакет підтримки текстів у кодуванні UTF-8 pdflatex
\RequirePackage{fontspec}                               % основний мовний пакет lua/xe-latex
\RequirePackage{polyglossia}                            % пакет для підтримки кирилічних шрифтів lua/xe-latex
\RequirePackage[autostyle]{csquotes}                    % пакет для поєднання декількох в документі

\setmainlanguage{ukrainian}                             % задання основної мови
\setotherlanguage{english}                              % задання допоміжної мови

\RequirePackage{amssymb,amsfonts,amsmath,amsthm,amscd}  % потрібні символи в режимі математичних формул

\defaultfontfeatures{Ligatures=TeX}                           % додаємо стандарт TeX ligatures
\setmainfont{Times New Roman}                                 % встановлюємо основний шрифт to Times New Roman
\newfontfamily\cyrillicfont{Times New Roman}[Script=Cyrillic] % визначаємо шрифт для кирилиці
\newfontfamily\cyrillicfonttt{Courier New}                    % встановлюємо моношрифт Courier New замість стандартного CMU Typewriter Text

\RequirePackage{upgreek}                                % пакет для прямих грецьких літер
\RequirePackage{textcomp}                               % потрібні символи в режимі математичних формул
\newcommand{\No}{\textnumero}                           % перевизначення символу "№"
\RequirePackage{mathtext}                               % потрібні символи в режимі математичних формул

% пакети для форматування відступів та полів
\RequirePackage[includehead, headsep=0pt, footskip=0pt, top=2cm, bottom=2cm, left=2cm, right=1cm]{geometry} 

\RequirePackage{indentfirst}                            % відступ для першого рядка абзацу
\RequirePackage{hanging}                                % відступ для всіх інших рядків абзацу
\RequirePackage[shortlabels]{enumitem}                  % пакет для форматування списків
\RequirePackage{setspace}                               % пакет для задання міжрядкового інтервалу
\RequirePackage{fancyhdr}                               % пакет для виставлення позиції номеру сторінки
\RequirePackage{etoolbox}                               % пакет для прибирання висячого відступу в бібліографії та прописування умовних відстаней між заголовками

% графічні пакети та таблиці
\RequirePackage{graphicx}               % основний пакет для вставки графіки
\RequirePackage{wrapfig}                % пакет для графіки, розміщеної між рядками тексту
\RequirePackage{float}                  % пакет для розширеного керування розміщенням плаваючих об'єктів
% \RequirePackage{subfig}                 % пакет для вставки підрисунків
\RequirePackage{subcaption}             % пакет для вставки підрисунків (subfig та subcaption конфліктують між собою)

\RequirePackage{flafter}                        % пакет гарантує, що рисунок чи таблиця ніколи не з'являться раніше посилання на самих себе
\RequirePackage[singlelinecheck=off]{caption}   % пакет для персоналізації заголовків рисунків та таблиць 
                                                % (вимкнена перевірка на те, щоб заголовок був у один рядок)

%\RequirePackage{xtab}                   % пакет для довгих таблиць
%\RequirePackage{longtable}              % пакет для довгих таблиць
\RequirePackage{multirow}                % пакет для об'єднання рядків таблиці
%\RequirePackage{array}                  % пакет для вирівнювання всередині таблиці
\RequirePackage{tabularx}                % пакет для просунутих таблиць
\RequirePackage{tabularray}              % пакет для ще більщ просунутих таблиць

% стилі заголовків та змісту
\RequirePackage[rm,small,raggedright]{titlesec}         % пакет для ручного керування стилем заголовків
\RequirePackage[titles]{tocloft}                        % пакет для ручного керування стилем змісту

\RequirePackage{url}                    % пакет для коректного представлення інтернет-посилань
\RequirePackage[hidelinks]{hyperref}    % пакет для гіперпосилань у змісті

\RequirePackage{icomma}                 % пакет для нормальних пробілів після ком у десяткових дробах
\RequirePackage{microtype}              % пакет для мікро-типографічних технік (просунуте вирівнювання рядків і переносів)

% пакети креслення засобами LaTeX
\RequirePackage{tikz}                   % основний пакет для креслення
\usetikzlibrary{arrows.meta}            % пакет для просунутих стрілок та вказівників

\RequirePackage{pgfplots}               % пакет для креслення графіків
\RequirePackage{pgfplotstable}          % пакет для завантаження даних у вигляді таблиць
\pgfplotsset{compat=newest}

% ------------------------------------------------------------- %
%  4. Основна частина
% ------------------------------------------------------------- %

% ------------------------------------------------------------- %
%  4.1 Початкові налаштування
% ------------------------------------------------------------- %

\sloppy % позбавляємось від переповнень
\linespread{1.5} % міжрядковий інтервал "1.5"

\newcolumntype{C}[1]{>{\centering\arraybackslash\setlength{\baselineskip}{0.8\baselineskip}}p{#1}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash\setlength{\baselineskip}{0.8\baselineskip}}p{#1}}

\newcommand{\setfontsize}[1]{\fontsize{#1pt}{#1pt}\selectfont}  % команда для задавання розміру шрифта
\newcommand\quotes[1]{\guillemotleft#1\guillemotright} % команда для французьких кутових лапок

\urlstyle{same} % текст гіперпосилань буде того ж шрифту, що й основний текст

% стиль сторінки для позиції номеру сторінки
\fancypagestyle{plain}{
    \fancyhf{}
    \rhead{\thepage}
        \renewcommand{\headrulewidth}{0pt} % видаляємо смугу хедера
        \renewcommand{\footrulewidth}{0pt} % видаляємо смугу футера
}

% команди для заборони використання перенесень
\tolerance=1
\emergencystretch=\maxdimen
\hyphenpenalty=10000
\hbadness=10000

\setlength{\parindent}{1.25cm} % задаємо відступ першого рядка абзацу (5 знаків, або 1.25 см)

% пропуск рядка до та після формули
\setlength{\belowdisplayskip}{21pt}
\setlength{\abovedisplayskip}{21pt}
\setlength{\belowdisplayshortskip}{21pt}
\setlength{\abovedisplayshortskip}{21pt}

% ------------------------------------------------------------- %
%  4.2 Оформлення списків, таблиць, рисунків
% ------------------------------------------------------------- %

% задамо оновлені списки, з правильними відступами
\AddEnumerateCounter{\asbuk}{\@asbuk}{\cyrm}
\setlist[itemize,1]{label=--, partopsep=0pt, parsep=0pt, topsep=0pt, itemsep=0pt, itemindent=1.7cm, leftmargin=0pt}
\setlist[itemize,2]{label=--, partopsep=0pt, parsep=0pt, topsep=0pt, itemsep=0pt, itemindent=2.25cm, leftmargin=0pt}
\setlist[enumerate,1]{
  label=\asbuk*) ,           
  partopsep=0pt, parsep=0pt,
  topsep=0pt, itemsep=0pt,
  leftmargin=0pt,
  itemindent=1.9cm,         
  labelsep=0.5em           
}

\setlist[enumerate,2]{
  label=\arabic*) ,       
  partopsep=0pt, parsep=0pt,
  topsep=0pt, itemsep=0pt,
  leftmargin=0pt,
  itemindent=2.6cm,      
  labelsep=0.5em
}

\setlist[enumerate,3]{
  label=--,    
  partopsep=0pt, parsep=0pt,
  topsep=0pt, itemsep=0pt,
  leftmargin=0pt,
  itemindent=3.3cm,    
  labelsep=0.5em
}

\setlist[enumerate,4]{
  label=--,    
  partopsep=0pt, parsep=0pt,
  topsep=0pt, itemsep=0pt,
  leftmargin=0pt,
  itemindent=4.0cm,    
  labelsep=0.5em
}

% заголовки таблиць, рисунків
\DeclareCaptionLabelFormat{tablecap}{\hspace*{1.25cm}Таблиця #2}        % формат за промовчанням "Табл." не влаштовує
\DeclareCaptionLabelFormat{figurecap}{Рисунок #2}                       % формат за промовчанням "Рис." не влаштовує
\DeclareCaptionFormat{listing}{Лістинг файлу #3}                        % формат за промовчанням "Listing" не влаштовує

\renewcommand{\thesubfigure}{\asbuk{subfigure}}

\captionsetup[table]{margin=0pt, singlelinecheck=false, justification=justified, labelsep=endash, labelformat=tablecap, font = {stretch = 1.5}} % формат для таблиць
\captionsetup[figure]{justification=centering, labelsep=endash, labelformat=figurecap, font = {stretch = 1.5}} % формат для рисунків
\captionsetup[lstlisting]{margin=0pt, format=listing, singlelinecheck=false, font = {stretch = 1.5}} % формат для лістингів

\setlength{\textfloatsep}{21pt}
\setlength{\intextsep}{21pt}

% визначаємо нове оточення для довгих таблиць
\DeclareTblrTemplate{middlehead, firstfoot, middlefoot, lastfoot}{dstu}{} % вимикаємо зайві сутності

% налаштовуємо стиль для заголовка - такий же, як і стиль звичайних таблиць
\DeclareTblrTemplate{firsthead}{dstu}{%
    \addtocounter{table}{-1}%
    \captionof{table}{\InsertTblrText{caption}}%
}

% налаштовуємо стиль для заголовка продовження таблиці
\DeclareTblrTemplate{lasthead}{dstu}{Продовження таблиці \thetable \vspace{0.3cm}}

% застосовуємо стиль
\SetTblrTemplate{firsthead, middlehead, lasthead, firstfoot, middlefoot, lastfoot}{dstu}

% додаємо вертикальний відступ від кінця таблиці до тексту після неї
\AtEndEnvironment{longtblr}{\vspace{0.5cm}}

% ------------------------------------------------------------- %
%  4.3 Визначення математичних оточень та роботи з формулами
% ------------------------------------------------------------- %

\theoremstyle{plain}
\newtheorem{theorem}{\indent \normalfont Теорема}[chapter]
\newtheorem{claim}{\indent \normalfont Твердження}[chapter]
\newtheorem{lemma}{\indent \normalfont Лема}[chapter]
\newtheorem{corollary}{\indent \normalfont Наслідок}[chapter]

\theoremstyle{definition}
\newtheorem{definition}{\indent \normalfont Визначення}[chapter]
\newtheorem{algorithm}{\indent \normalfont Алгоритм}[chapter]
\newtheorem{problem}{\indent \normalfont Задача}[chapter]
\newtheorem{example}{\indent \normalfont Приклад}[chapter]

\theoremstyle{remark}
\newtheorem*{remark}{\indent \normalfont Зауваження}

\renewenvironment{proof}{{\indent Доведення.~}}{\qed}

% змінюємо формат нумерації формул
\renewcommand{\theequation}{\arabic{chapter}.\arabic{equation}}

% команди для заборони розриву формул у небажаному місці
\binoppenalty=\maxdimen
\relpenalty=\maxdimen

% виправляємо помилки з апострофами в формулах
\makeatletter
\def\latex@@rquote{^\bgroup\prim@s} 
\let\hang@@h@ngrquote\h@ngrquote
\renewcommand{\h@ngrquote}{
        \relax\ifmmode
                \expandafter\latex@@rquote
        \else
                \expandafter\hang@@h@ngrquote
        \fi}
\makeatother

% ------------------------------------------------------------- %
%  4.4 Ненумеровані розділи
% ------------------------------------------------------------- %

% команда для створення ненумерованого розділу, який тим не менше показується в змісті
\newcommand{\uchapter}[1]{\chapter*{#1}\addcontentsline{toc}{chapter}{#1}} 

\newcommand{\intro}{\uchapter{Вступ}}                   % для створення вступу використаємо команду \intro
\newcommand{\conclusions}{\uchapter{Висновки}}          % для створення висновків використаємо команду \conslusions
\newcommand{\shortings}{\uchapter{Перелік скорочень, умовних позначень, термінів}}
                                                        % для створення переліку умовних позначень використаємо команду \shortings

\newcommand{\chapconclude}{\section{Висновки до розділу}}

% нові команди для анотацій (двома мовами)
\newcommand{\abstractUkr}{\chapter*{АНОТАЦІЯ}\thispagestyle{empty}}
\newcommand{\abstractEng}{\chapter*{ABSTRACT}\thispagestyle{empty}}

% ------------------------------------------------------------- %
%  4.5 Заголовки розділів
% ------------------------------------------------------------- %

% спочатку --- для розділу
\titleformat{\chapter} 				             % указуємо, що модифікуємо саме розділ
                {\centering\normalsize\MakeUppercase}        % указуємо формат назви ("усі великі")
                {\hspace{1cm}\thechapter}             	     % указуємо формат власне номера: це буде просто число, без крапки
                {0.5em}                            	     % відстань між номером і назвою
                {}                        		     % текст, що передує назві
        
% тепер --- для підрозділу
\titleformat{\section}[block] 			             % указуємо, що модифікуємо саме підрозділ
                {\normalsize}                                % указуємо формат назви
                {\hspace{1.25cm}\thesection}  		     % указуємо формат власне номера: це буде просто число, без крапки
                {0.5em}                               	     % відстань між номером і назвою
                {}                             		     % текст, що передує назві

% тепер --- для підрозділу без нумерації
\titleformat{name=\section,numberless}[block] 		     % указуємо, що модифікуємо саме підрозділ
                {\normalsize}                                % указуємо формат назви
                {\hspace{1.25cm}}  			     % указуємо формат власне номера: це буде просто число, без крапки
                {0.0em}                               	     % відстань між номером і назвою
                {}                             		     % текст, що передує назві

% тепер --- для підрозділу ще нижчого рівня
\titleformat{\subsection}[block]                    % указуємо, що модифікуємо саме підрозділ нижнього рівня
                {\normalsize}                       % указуємо формат назви
                {\hspace{1.25cm}\thesubsection}     % указуємо формат власне номера: це буде просто число, без крапки
                {0.5em}                             % відстань між номером і назвою
                {}                             	    % текст, що передує назві
                        
% тепер задамо відступи для заголовку розділу
\titlespacing{\chapter} % указуємо, що модифікуємо саме розділ
                {0pt}           % відступ зліва
                {0pt}           % відступ зверху (немає)
                {48pt}          % відступ знизу (2 рядки)

% тепер задамо відступи для заголовку підрозділу                        
\titlespacing{\section} % указуємо, що модифікуємо саме підрозділ
                {0em}           % відступ зліва
                {42pt}          % відступ зверху (2 рядки)
                {42pt}          % відступ знизу (2 рядки)

%% тепер задамо відступи для заголовку підрозділу без нумерації                  
\titlespacing{name=\section,numberless} 
				{0em}           % відступ зліва
				{42pt}          % відступ зверху (2 рядки)
				{42pt}          % відступ знизу (2 рядки)
     
% тепер задамо відступи для заголовку пункта
\titlespacing{\subsection} % указуємо, що модифікуємо саме пункт
				{0em}           % відступ зліва
				{42pt}          % відступ зверху (2 рядки)
				{42pt}          % відступ знизу (2 рядки)

% тепер зробимо так, щоб між двома заголовками була звичайна відстань
\preto\section{%
        \ifnum\value{section}=0\addvspace{-42pt}\fi%
}

\preto\subsection{%
        \ifnum\value{subsection}=0\addvspace{-42pt}\fi%
}

% ------------------------------------------------------------- %
% 4.6 Додатки
% ------------------------------------------------------------- %

% хитра команда для створення додатків
\newcounter{appendnum}
\newcommand{\append}[1]{
	\stepcounter{appendnum}
	\titleformat{\chapter}{\centering\normalsize}{\hspace{1cm}\thechapter}{0.5em}{}
	\chapter*{Додаток \Asbuk{appendnum}\\
	#1}
	\addcontentsline{toc}{chapter}{Додаток \Asbuk{appendnum}\ #1}
	\renewcommand{\thefigure}{\Asbuk{appendnum}.\arabic{figure}}
	\renewcommand{\thetable}{\Asbuk{appendnum}.\arabic{table}}
	\renewcommand{\thechapter}{\Asbuk{appendnum}}
	\renewcommand{\thesection}{\Asbuk{appendnum}.\arabic{section}}
	\renewcommand{\thesubsection}{\Asbuk{appendnum}.\arabic{section}.\arabic{subsection}}
	\setcounter{section}{0}
	\setcounter{subsection}{0}
	\setcounter{figure}{0}
	\setcounter{table}{0}
}

% ------------------------------------------------------------- %
%  4.7 Зміст
% ------------------------------------------------------------- %

% видозмінюємо зміст
\makeatletter
\renewcommand{\tableofcontents}{
        \clearpage
        \thispagestyle{plain}
        \begin{center}
            ЗМІСТ
        \end{center}

        \addvspace{42pt}
        \@starttoc{toc}
}
\makeatother

% задаємо відстань між точками
\renewcommand{\cftchapdotsep}{1}
\renewcommand{\cftsecdotsep}{1}
\renewcommand{\cftsubsecdotsep}{1}

% задаємо розмір поля, що виділяється під номер сторінки
\cftsetpnumwidth{1em}

% задамо відстані між рядками в змісті
\setlength\cftbeforechapskip{0pt}

% займемося розділами
\renewcommand{\cftchapaftersnum}{}              % що буде писатися після номеру розділу (нічого)
\renewcommand{\cftchapindent}{0cm}              % відступ номера розділу від лівого краю
\renewcommand{\cftchapnumwidth}{1em}            % розмір поля, що виділяється під номер розділу
\renewcommand{\cftchappagefont}{\mdseries}      % стиль номеру сторінки (звичайний, 14pt)
\renewcommand{\cftchapfont}{\mdseries}          % стиль заголовку (звичайний)
\renewcommand{\cftchapleader}{\cftdotfill{\cftchapdotsep}} % чим заповнювати проміжок від заголовку до номеру сторінки (точками)

% тепер займемося підрозділами
\renewcommand{\cftsecaftersnum}{}               % що буде писатися після номеру підрозділу (нічого)
\renewcommand{\cftsecindent}{1em}               % відступ номера підрозділу від лівого краю
\renewcommand{\cftsecnumwidth}{2em}             % розмір поля, що виділяється під номер підрозділу
\renewcommand{\cftsecpagefont}{\mdseries}       % стиль номеру сторінки (звичайний, 14pt)
\renewcommand{\cftsecfont}{\mdseries}           % стиль заголовку (звичайний)
\renewcommand{\cftsecleader}{\cftdotfill{\cftsecdotsep}} % чим заповнювати проміжок від заголовку до номеру сторінки (точками)

% тепер займемося пунктами
\renewcommand{\cftsubsecaftersnum}{}            % що буде писатися після номеру пункта (нічого)
\renewcommand{\cftsubsecindent}{3em}            % відступ номера пункта від лівого краю
\renewcommand{\cftsubsecnumwidth}{2.5em}        % розмір поля, що виділяється під номер пункта
\renewcommand{\cftsubsecpagefont}{\mdseries}    % стиль номеру сторінки (звичайний, 14pt)
\renewcommand{\cftsubsecfont}{\mdseries}        % стиль заголовку (звичайний)
\renewcommand{\cftsubsecleader}{\cftdotfill{\cftsubsecdotsep}} % чим заповнювати проміжок від заголовку до номеру сторінки (точками)

% ------------------------------------------------------------- %
%  5. Бібліографія
% ------------------------------------------------------------- %

\makeatletter
\renewcommand{\thebibliography}[1]{%
    \chapter*{\MakeUppercase{Перелік посилань}} % змінюємо заголовок
	\addcontentsline{toc}{chapter}{Перелік посилань} % додаємо в зміст
	\list{\arabic{enumiv}.}{% змінюємо формат номеру для переліку посилань (число + крапка)
		% далі наведемо порядок із відступами
                \setlength{\itemindent}{1.9cm} % відступ номера посилання на 1.25cm
                \setlength{\leftmargin}{0pt} % відступ тіла посилання
		\setlength{\partopsep}{0pt}%
		\setlength{\parskip}{0pt}%
		\setlength{\parsep}{0pt}%
		\setlength{\topsep}{0pt}%
		\setlength{\itemsep}{0pt}%
		\usecounter{enumiv}%
	}%
	\sloppy
	\clubpenalty4000
	\widowpenalty4000
	\sfcode`\.\@m
}
\makeatother

\newcommand*{\DSTUAllowNamesAnd}{0} % забороняємо "та" у переліку авторів 
\newcommand*{\DSTUResponsibilityName}[2]{#2} % зворотній порядок ініціали/ім'я

\ProvideTextCommandDefault{\cyrdash}{\textendash} % перезаписуємо формат символу дефіса
\newcommand*{\DSTUAllowBibDash}{1} % увімкнути '1' чи вимкнути '0' дефіси у бібліографії

% Коли у тексті не наведено жодного цитування, виникає помилка компіляції bibtex:
% I found no \citation commands---while reading file thesis.aux
% Костиль - за відсутності цитувань компілюватимемо деяке бібліографічне джерело за замовчуванням
\RequirePackage{etoolbox} % допоміжна бібліотека для підрахунку цитувань у тексті

\newbool{havecites}
\setbool{havecites}{false}

\pretocmd{\cite}{\setbool{havecites}{true}}{}{}
\pretocmd{\citep}{\setbool{havecites}{true}}{}{}
\pretocmd{\citet}{\setbool{havecites}{true}}{}{}

\AtEndDocument{
	\ifbool{havecites}
		{} % якщо у тексті є бодай одне цитування - не втручаємося
		{\nocite{*}} % якщо у тексті відсутні цитування - цитуємо всі позиції у бібліографії (зірочка *)
}

% пакет для використання бібліографії засобами biblatex, а не bibtex
% \RequirePackage[
% 	backend = biber,
% 	style = numeric,
% 	sorting = none,
% 	minalphanames = 3,
% 	maxbibnames = 5
% ]{biblatex}
% \addbibresource{thesis.bib}

% ------------------------------------------------------------- %
%  6. Лістинг програм
% ------------------------------------------------------------- %

\RequirePackage{listings}               % пакет для відображення форматування вихідних кодів
\RequirePackage{verbatim}               % пакет для кодів програм
\RequirePackage{xcolor}                 % пакет для кольорового забарвлення коду

\definecolor{lightgray}{rgb}{0.95,0.95,0.95}
\definecolor{deepblue}{rgb}{0,0,0.5}
\definecolor{deepred}{rgb}{0.6,0,0}
\definecolor{deepgreen}{rgb}{0,0.5,0}

\colorlet{commentcolour}{green!50!black}
\colorlet{stringcolour}{red!60!black}
\colorlet{keywordcolour}{blue!90!black}
\colorlet{keywordcolour2}{magenta!80!black}
\colorlet{commandcolour}{blue!60!black}
\colorlet{numpycolour}{yellow!60!black}
\colorlet{exceptioncolour}{yellow!50!red}
\colorlet{literatecolour}{gray!80!white}
\colorlet{promptcolour}{green!50!black}
\colorlet{specmethodcolour}{violet}

\lstdefinestyle{dstustyle}{
    language         = python,
    frame            = none, % single
    keepspaces       = true,
    % numbers          = left,
    % numberstyle      = \tiny,
    % numbersep        = 5pt,
    showtabs         = false,
    tabsize          = 4,
    basicstyle       = \linespread{0.66}\tiny\ttfamily,
    stringstyle      = \color{stringcolour},
    showstringspaces = false,
    upquote          = true,
    morecomment      = [s]{"""}{"""},
    commentstyle     = \color{commentcolour}\slshape,
    rulecolor        = \color{black!40},
    backgroundcolor  = {},
    breakindent      = .5\textwidth,
    breaklines       = true,
    prebreak         = \raisebox{0ex}[0ex][0ex]{\ensuremath{\hookleftarrow}},
    % otherkeywords    = {\%, \}, \{, \&, \|},
    alsoletter       = {1234567890},
    keywordstyle     = \color{keywordcolour}\bfseries,
    emph             = {[1]and, class, def, global, in, lambda, not, or, pass, print, raise, try, assert, with},
    emphstyle        = {[1]\color{keywordcolour}\bfseries},
    emph             = {[2]True, False, None},
    emphstyle        = {[2]\color{keywordcolour}},
    emph             = {[3]object, type, isinstance, copy, deepcopy, zip, enumerate, reversed, list, set, dict, tuple, xrange, append, add, execfile, real, imag, reduce, str, repr},
    emphstyle        = {[3]\color{commandcolour}},
    emph             = {[4]Exception, NameError, IndexError, SyntaxError, TypeError, ValueError, OverflowError, ZeroDivisionError},
    emphstyle        = {[4]\color{exceptioncolour}\bfseries},
    emph             = {[5]range, ode, fsolve, sqrt, exp, sin, cos,arctan, arctan2, arccos, pi,  array, norm, solve, dot, range, len, max, sum, flatten, shape, reshape, find, any, all, abs, plot, linspace, legend, quad, polyval, polyfit, hstack, concatenate, vstack, column_stack, empty, zeros, ones, rand, vander, grid, pcolor, eig, eigs, eigvals, svd, qr, tan, det, logspace, roll, min, mean, cumsum, cumprod, diff, vectorize, lstsq, cla, eye, xlabel, ylabel, squeeze},
    emphstyle        = {[5]\color{numpycolour}},
    emph             = {[6]__init__,__add__,__mul__,__div__,__sub__,__call__,__getitem__,__setitem__,__eq__,__ne__,__nonzero__,__rmul__,__radd__,__repr__,__str__,__get__,__truediv__,__pow__,__name__,__future__,__all__},
    emphstyle        = {[6]\color{specmethodcolour}},
    emph             = {[7]break, continue, yield, del, elif, else, except, exec, finally, for, from, if, import, return},
    emphstyle        = {[7]\color{keywordcolour2}},
    literate         =
	{а}{\cyra}1
	{б}{{\selectfont\char225}}1
	{в}{{\selectfont\char226}}1
	{г}{{\selectfont\char227}}1
	{д}{{\selectfont\char228}}1
	{е}{{\selectfont\char229}}1
	{ж}{{\selectfont\char230}}1
	{з}{{\selectfont\char231}}1
	{и}{{\selectfont\char232}}1
	{й}{{\selectfont\char233}}1
	{к}{{\selectfont\char234}}1
	{л}{{\selectfont\char235}}1
	{м}{{\selectfont\char236}}1
	{н}{{\selectfont\char237}}1
	{о}{{\selectfont\char238}}1
	{п}{{\selectfont\char239}}1
	{р}{{\selectfont\char240}}1
	{с}{{\selectfont\char241}}1
	{т}{{\selectfont\char242}}1
	{у}{{\selectfont\char243}}1
	{ф}{{\selectfont\char244}}1
	{х}{{\selectfont\char245}}1
	{ц}{{\selectfont\char246}}1
	{ч}{{\selectfont\char247}}1
	{ш}{{\selectfont\char248}}1
	{щ}{{\selectfont\char249}}1
	{ь}{{\selectfont\char252}}1
	{ю}{{\selectfont\char254}}1
	{я}{{\selectfont\char255}}1
	{А}{{\selectfont\char192}}1
	{Б}{{\selectfont\char193}}1
	{В}{{\selectfont\char194}}1
	{Г}{{\selectfont\char195}}1
	{Д}{{\selectfont\char196}}1
	{Е}{{\selectfont\char197}}1
	{Ж}{{\selectfont\char198}}1
	{З}{{\selectfont\char199}}1
	{И}{{\selectfont\char200}}1
	{Й}{{\selectfont\char201}}1
	{К}{{\selectfont\char202}}1
	{Л}{{\selectfont\char203}}1
	{М}{{\selectfont\char204}}1
	{Н}{{\selectfont\char205}}1
	{О}{{\selectfont\char206}}1
	{П}{{\selectfont\char207}}1
	{Р}{{\selectfont\char208}}1
	{С}{{\selectfont\char209}}1
	{Т}{{\selectfont\char210}}1
	{У}{{\selectfont\char211}}1
	{Ф}{{\selectfont\char212}}1
	{Х}{{\selectfont\char213}}1
	{Ц}{{\selectfont\char214}}1
	{Ч}{{\selectfont\char215}}1
	{Ш}{{\selectfont\char216}}1
	{Щ}{{\selectfont\char217}}1
	{Ь}{{\selectfont\char220}}1
	{Ю}{{\selectfont\char222}}1
	{Я}{{\selectfont\char223}}1
	{і}{ {\selectfont\char105} }1
	{ї}{ {\selectfont\char168} }1
	{є}{ {\selectfont\char185} }1
	{ґ}{ {\selectfont\char160} }1
	{І}{ {\selectfont\char73} }1
	{Ї}{ {\selectfont\char136} }1
	{Є}{ {\selectfont\char153} }1
	{Ґ}{ {\selectfont\char128} }1       
}
