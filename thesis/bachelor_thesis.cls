% ********** Клас для написання пояснювальної записки **********
% *********  до атестаційної роботи ступеня бакалавра **********
% Зміни Таврова Д. Ю. в шаблоні, створеному Яковлєвим С. В. (ММЗІ)
% на основі шаблону, створеного Тавровим Д. Ю., Борисенком П. Б. та Дутчаком І. О. (ПМА)

% 1. Ідентифікуємо клас
\ProvidesClass{bachelor_thesis}[2024/05/01 v0.6 -- Пояснювальна записка до кваліфікаційної роботи бакалавра]
\NeedsTeXFormat{LaTeX2e} % потрібний формат


% 2. Завантажуємо "батьківський клас"
\LoadClass[a4paper,14pt,ukrainian,oneside]{extreport}


% 3. Завантажуємо пакети

% мовні пакети
\RequirePackage{cmap}                                   % для кодування шрифтів у pdf
%\RequirePackage[T2A]{fontenc}                           % пакет для підтримки кирилічних шрифтів
%\RequirePackage[utf8]{inputenc}                         % пакет для підтримки текстів у кодуванні UTF-8
\RequirePackage{fontspec}
\RequirePackage{polyglossia}         % пакет для підтримки української та російської мов
\RequirePackage{amssymb,amsfonts,amsmath,amsthm,amscd}  % потрібні символи в режимі математичних формул
\RequirePackage[backend=bibtex,style=numeric]{biblatex} % пакет BibLatex для коректної бібліографії

\setmainlanguage{ukrainian}  % Set main language to Ukrainian
\setotherlanguage{english}  % Set other language to English

\defaultfontfeatures{Ligatures=TeX}  % Enable standard TeX ligatures
\setmainfont{Times New Roman}  % Set main font to Times New Roman
\newfontfamily\cyrillicfont{Times New Roman}[Script=Cyrillic]  % Specify font for Cyrillic script
\newfontfamily\cyrillicfonttt{Courier New}

\RequirePackage{upgreek}                                % пакет для прямих грецьких літер
\RequirePackage{textcomp}                               % потрібні символи в режимі математичних формул
\RequirePackage{mathtext}                               % потрібні символи в режимі математичних формул

% пакети для форматування відступів та полів
\RequirePackage[includehead, headsep=0pt, footskip=0pt, top=2cm, bottom=2cm, left=2cm, right=1cm]{geometry} 
                                                        % задаємо поля сторінки

\RequirePackage{indentfirst}                            % відступ для першого рядка абзацу
\RequirePackage{hanging}                                % відступ для всіх інших рядків абзацу
\RequirePackage[shortlabels]{enumitem}                  % пакет для форматування списків
\RequirePackage{setspace}                               % пакет для того, щоб можна було задати міжрядковий інтервал
\RequirePackage{fancyhdr}                               % пакет потрібний для виставлення позиції номеру сторінки
\RequirePackage{etoolbox}                               % пакет для прибирання висячого відступу в бібліографії та прописування умовних відстаней між заголовками


% графічні пакети та таблиці
\RequirePackage{graphicx}               % основний пакет для вставки графіки
%\RequirePackage{subfig}                 % пакет для вставки підрисунків
\RequirePackage{subcaption}             % пакет для вставки підрисунків (subfig та subcaption конфліктують між собою)

\RequirePackage{flafter}                        % пакет гарантує, що рисунок чи таблиця ніколи не з'являться раніше посилання на самих себе
\RequirePackage[singlelinecheck=off]{caption}   % пакет для персоналізації заголовків рисунків та таблиць 
                                                % (вимкнена перевірка на те, щоб заголовок був у один рядок

%\RequirePackage{xtab}                   % пакет для довгих таблиць
%\RequirePackage{longtable}              % пакет для довгих таблиць
\RequirePackage{multirow}                % пакет для об'єднання рядків таблиці
%\RequirePackage{array}                  % пакет для вирівнювання всередині таблиці
\RequirePackage{tabularx}                % пакет для просунутих таблиць

% стилі заголовків та змісту
\RequirePackage[rm,small,raggedright]{titlesec}         % пакет для ручного керування стилем заголовків
%\RequirePackage[titles,subfigure]{tocloft}              % пакет для ручного керування стилем змісту
\RequirePackage[titles]{tocloft}                        % пакет для ручного керування стилем змісту

\RequirePackage{url}                    % пакет для коректного представлення інтернет-посилань у "Переліку посилань"
\RequirePackage[hidelinks]{hyperref}    % пакет для гіперпосилань у змісті

\RequirePackage{listings}               % пакет для відображення форматування вихідних кодів
\RequirePackage{verbatim}               % пакет для кодів програм

%\RequirePackage{xkeyval}                % пакет для параметричних команд (де я їх використовую?)

\RequirePackage{icomma}                 % пакет для нормальних пробілів після ком у десяткових дробах
\RequirePackage{csquotes}               % для коректної роботи biblatex

% ************************************************************* %
%  4. Основна частина
% ************************************************************* %

% початкові налаштування

\sloppy                 % Позбавляємось від переповнень
\linespread{1.5} % інтервал "1,5"

\newcolumntype{C}[1]{>{\centering\arraybackslash\setlength{\baselineskip}{0.8\baselineskip}}p{#1}}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash\setlength{\baselineskip}{0.8\baselineskip}}p{#1}}

\newcommand{\setfontsize}[1]{\fontsize{#1pt}{#1pt}\selectfont}  % команда для задавання розміру шрифта
\newcommand\quotes[1]{\guillemotleft#1\guillemotright} % команда для французьких кутових лапок

\urlstyle{same}         % текст гіперпосилань буде того ж шрифту, що й основний текст

% стиль сторінки для позиції номеру сторінки
\fancypagestyle{plain}{
    \fancyhf{}
    \rhead{\thepage}
            \renewcommand{\headrulewidth}{0pt}
            \renewcommand{\footrulewidth}{0pt}
}

% команди для заборони використання перенесень
\tolerance=1
\emergencystretch=\maxdimen
\hyphenpenalty=10000
\hbadness=10000

\setlength{\parindent}{1.25cm} % задаємо відступ першого рядка абзацу (5 знаків, або 1,25 см)

% пропуск рядка до та після формули
\setlength{\belowdisplayskip}{21pt}
\setlength{\abovedisplayskip}{21pt}
\setlength{\belowdisplayshortskip}{21pt}
\setlength{\abovedisplayshortskip}{21pt}

% ************************************************************* %
%  Оформлення списків, таблиць, рисунків
% ************************************************************* %

% задамо оновлені списки, з правильними відступами
\AddEnumerateCounter{\asbuk}{\@asbuk}{\cyrm}
\setlist[itemize,1]{label=--, partopsep=0pt, parsep=0pt, topsep=0pt, itemsep=0pt, itemindent=1.7cm, leftmargin=0pt}
\setlist[itemize,2]{label=--, partopsep=0pt, parsep=0pt, topsep=0pt, itemsep=0pt, itemindent=2.25cm, leftmargin=0pt}
\setlist[enumerate,1]{label=\arabic*), partopsep=0pt, parsep=0pt, topsep=0pt, itemsep=0pt, itemindent=1.9cm, leftmargin=0pt}
\setlist[enumerate,2]{label=\asbuk*),  partopsep=0pt, parsep=0pt, topsep=0pt, itemsep=0pt, itemindent=2.6cm, leftmargin=0pt}

% заголовки таблиць, рисунків
\DeclareCaptionLabelFormat{tablecap}{\hspace*{1.25cm}Таблиця #2}        % формат за промовчанням "Табл." не влаштовує
\DeclareCaptionLabelFormat{figurecap}{Рисунок #2}                       % формат за промовчанням "Рис." не влаштовує
\DeclareCaptionFormat{listing}{Лістинг файлу #3} % формат за промовчанням "Listing" не влаштовує

\renewcommand{\thesubfigure}{\asbuk{subfigure}}

\captionsetup[table]{margin=0pt, singlelinecheck=false, justification=justified, labelsep=endash, labelformat=tablecap, font = {stretch = 1.5}} % формат для таблиць
\captionsetup[figure]{justification=centering, labelsep=endash, labelformat=figurecap, font = {stretch = 1.5}} % формат для рисунків
\captionsetup[lstlisting]{margin=0pt, format=listing, singlelinecheck=false, font = {stretch = 1.5}} % формат для лістингів

\setlength{\textfloatsep}{21pt}
\setlength{\intextsep}{21pt}

% нове оточення для довгих таблиць
\newenvironment{ltable}[4]{
        \begingroup\centering
        \setlength{\LTpre}{-14pt}
        \setlength{\LTpost}{14pt}
        \setlength{\LTcapwidth}{0pt}
        \begin{longtable}{#1}
                \kill
                \caption{#2}\label{#3}\\
                #4
                \endfirsthead
                \kill
                \caption*{Продовження таблиці \thetable}\\
                        #4
                        \endhead
                        \hline
                        \endfoot
        }
        {%
                \end{longtable}
                \endgroup       
        }


% ************************************************************* %
%  Визначення математичних оточень та роботи з формулами
% ************************************************************* %

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


% ************************************************************* %
%  Ненумеровані розділи
% ************************************************************* %

% команда для створення ненумерованого розділу, який тим не менше показується в змісті
\newcommand{\uchapter}[1]{\chapter*{#1}\addcontentsline{toc}{chapter}{#1}} 

\newcommand{\intro}{\uchapter{Вступ}}                   % для створення вступу використаємо команду \intro
\newcommand{\conclusions}{\uchapter{Висновки}}          % для створення висновків використаємо команду \conslusions
\newcommand{\shortings}{\uchapter{Перелік умовних позначень, скорочень і термінів}}
                                                        % для створення переліку умовних позначень використаємо команду \shortings

\newcommand{\chapconclude}{\section{Висновки до розділу}\addcontentsline{toc}{section}{Висновки до розділу}}

% нові команди для анотацій (трьома мовами)
\newcommand{\abstractUkr}{\chapter*{АНОТАЦІЯ}\thispagestyle{empty}\renewcommand{\arraystretch}{1.5}\setfontsize{14}}
\newcommand{\abstractEng}{\chapter*{ABSTRACT}\thispagestyle{empty}\renewcommand{\arraystretch}{1.5}\setfontsize{14}}


% ************************************************************* %
%  Заголовки розділів
% ************************************************************* %

% спочатку --- для розділу
\titleformat{\chapter} 										 % указуємо, що модифікуємо саме розділ
                {\centering\normalsize\MakeUppercase}        % указуємо формат назви ("усі великі")
                {\hspace{1cm}\thechapter}             	     % указуємо формат власне номера: це буде просто число, без крапки
                {0.5em}                            		     % відстань між номером і назвою
                {}                        				     % текст, що передує назві
        
% тепер --- для підрозділу
\titleformat{\section}[block] 								 % указуємо, що модифікуємо саме підрозділ
                {\normalsize}                                % указуємо формат назви
                {\hspace{1.25cm}\thesection}  				 % указуємо формат власне номера: це буде просто число, без крапки
                {0.5em}                               	     % відстань між номером і назвою
                {}                             				 % текст, що передує назві

% тепер --- для підрозділу ще нижчого рівня
\titleformat{\subsection}[block] % указуємо, що модифікуємо саме підрозділ нижнього рівня
				{\normalsize}                                % указуємо формат назви
				{\hspace{1.25cm}\thesubsection} 			 % указуємо формат власне номера: це буде просто число, без крапки
				{0.5em}                               	     % відстань між номером і назвою
				{}                             				 % текст, що передує назві
                        
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

% ************************************************************* %
%  Додатки
% ************************************************************* %

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

% ************************************************************* %
%  Зміст
% ************************************************************* %

% видозмінюємо зміст
\makeatletter
\renewcommand{\tableofcontents}{
        \clearpage
        \thispagestyle{plain}
        \begin{center}
            Зміст
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


% ************************************************************* %
%  Бібліографія
% ************************************************************* %


% наведемо порядок із відступами в бібліографії
\patchcmd{\thebibliography}
        {\advance\leftmargin\labelsep}
        {\leftmargin=0pt\itemindent=1.9cm\advance\itemindent\labelsep}
        {}
        {}

% змінюємо формат номеру для переліку посилань (число + крапка)
\makeatletter
\renewcommand{\@biblabel}[1]{#1.}
\makeatother

% оновимо стандартне оточення, оскільки потрібно додати перелік посилань у зміст
\let\tempbib\thebibliography
\let\tempbibend\endthebibliography
\renewenvironment{thebibliography}{
        \renewcommand{\bibname}{ПЕРЕЛІК ПОСИЛАНЬ}% змінюємо заголовок
        \tempbib{999.}           % максимально можливий номер джерела --- 999
        % далі наведемо порядок із відступами
        \setlength\partopsep{0pt}
        \setlength\parskip{0pt}
        \setlength\parsep{0pt}
        \setlength\topsep{0pt}
        \setlength\itemsep{0pt}
        \addcontentsline{toc}{chapter}{Перелік посилань}        % додаємо в зміст
}
{
        \tempbibend
}

% деякі стильові параметри лістингів програм
\RequirePackage{xcolor}
\definecolor{deepblue}{rgb}{0,0,0.5}
\definecolor{deepred}{rgb}{0.6,0,0}
\definecolor{deepgreen}{rgb}{0,0.5,0}
\lstloadlanguages{Python}%
\lstset{language=Python,                             % Use Python
	frame=none,                           % No frame around code
	basicstyle=\linespread{0.8}\footnotesize\ttfamily,             % Use small true type font
	keywordstyle=\color{deepblue},                          % R functions as usual
	keywordstyle=[2]\color{deepblue},           % R function arguments blue
	morekeywords={as},
	identifierstyle=,                       % Nothing special about identifiers
	commentstyle=\color{deepgreen},       % Comments are dark green
	stringstyle=\color{deepgreen},        % Strings are dark green
	numberstyle=\color{blue},               % Numbers are blue
	showstringspaces=false,                 % Don't put marks in string spaces
	tabsize=4,                              % 4 spaces per tab
	%
	%%% Put standard R functions not included in the default
	%%% language here
	%morekeywords={xlim,ylim,var,alpha,factorial,poissrnd,normpdf,normcdf},
	%
	%%% Put R function parameters here
	%morekeywords=[2]{on, off, interp},
	%
	extendedchars=true, %compare with false instead
	literate={а}{\cyra}1
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