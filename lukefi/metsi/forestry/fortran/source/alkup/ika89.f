*	PUIDEN RINNANKORKEUSIKAMALLIT - HELENA HENTTONEN 23.03.1989

*	- puuttuu - TARKEMPI MALLIEN, TOIMINNAN JA KAYTON SELITYS

*	HUOM: MALLI 1 TOIMII VAIN RELASKOOPPIKOEALOILLA, KOSKA 
*	SUHTEELLINEN KOKO LASKETAAN ANNETTUJEN LAPIMITTOJEN 
*	ARITMEETTISENA KESKIARVONA.

*	===================
*	KALIBROINTIOSITTEET
*	===================




*	======================================
*	MALLIEN KAYTTO KALIBROINTIOSITTEITTAIN
*	======================================


*	--------------
	FUNCTION F_AGE(TYP,TEM,DRA,DIS,S,D,F,CS,CD,CA,NC,ED,NE)
*	--------------

*	PAKOLLISET SELITTAJAT

*	kalibrointipuun ja tarkasteltavan puun kasvupaikkatunnukset

* 	TYP VMI7 metsatyyppiryhma
*	TEM lamposumma
*	DRA VMI7 ojitustilanne
*	DIS etaisyys rannikosta, - puuttuu selitys -
*	    jos DIS <= 0, oletetaan etaisyydeksi yli 2 km

*	tarkasteltavan puun tunnukset

*	S VMI7 puulaji
*	D lapimitta, cm
*	F VMI7 latvuskerros

*	MAHDOLLISET SELITTAJAT

*	kalibrointipuiden tunnukset

*	CS VMI7 puulaji
*	CD lapimita, cm
*	CA rinnankorkeusika, v
*	NC kalibrointipuiden maara CS:ssa, CD:ssa ja CA:ssa
*	   jos NC = 0, ei kalibroida

*	ED koealan kaikkien puiden lapimitat, cm
*	(myos tarkasteltava puu ja kalibrointipuu, jos se on koealalta)
*	NE puiden maara ED:ssa
*	jos NE = 0, kaytetaan mallilajia 2, jossa puun suhteellinen
*       koko ei ole selittajana

	PARAMETER (MXC=20)
	DIMENSION CSE(MXC),CDE(MXC),CAE(MXC)

	DIMENSION ED(*),CS(*),CD(*),CA(*)	


*	VALITSE MALLILAJI
*	-----------------

*	jos on annettu koealan puiden lapimitat, kaytetaan 
*	selittajana myos puun suhteellista kokoa
	IF(NE.GT.0)THEN
*	  malli, jossa puun suhteellinen koko on selittajana
	  LAJI=1
	ELSE
*	  malli, jossa puun suhteellinen koko ei ole selittajana
	  LAJI=2
	ENDIF


*	LASKE ENNUSTE METSIKKOTEKIJALLE
*	-------------------------------

*	JOS ON KALIBROINTITUNNUKSET ...
*	- enta jos kalibrointitiedot ovat kuitenkin nollia?
	IF(NC.GT.0)THEN
	  IF(NC.GT.MXC)THEN
	    WRITE(6,*)'-->> F_IKA: LIIKAA KALIBROINTIPUITA'
	    WRITE(6,*)'     AJOA EI KANNATA JATKAA.'
	    STOP
	  ENDIF
*	  ... SELVITA, OVATKO MALLILAJI JA KALIBROINTITUNNUKSET 
*	  MUUTTUNEET EDELLISESTA KUTSUSTA
*	  jos mallilaji on sama kuin edellisella kerralla ja ...
	  IF(LAJI.NE.LAJIE)GO TO 2
*           ... ja kaikki koealatiedot samoja
            IF(TYP.NE.TYPE.OR.TEM.NE.TEME.OR.DRAE.NE.DRA.OR.DIS.NE.DISE)
     -         GO TO 2
*	    ... ja kaikki kalibrointitiedot samoja ...
	    DO 1 I=1,NC
	      IF(CS(I).NE.CSE(I).OR.CD(I).NE.CDE(I).OR.CA(I).NE.CAE(I))
     -	        GO TO 2	    
1	    CONTINUE	  
*	  ... voidaan kayttaa entista kalibrointikerrointa
	  GO TO 3	
2	  CONTINUE
*	  muuten laske uusi kalibrointikerroin eli ...
*	  ... laske kalibrointipuiden ika malleilla ...
*         kalibrointipuiden ian ennustevirheiden summa
          ERO=0
          DO 12 I=1,NC
          IF(CA(I).LT.1)GO TO 12
	  IF(LAJI.EQ.1)THEN
*	    laske ensin kalibrointipuun suhteellinen koko selittajaksi
*	    suhteellinen koko = 
*	    lapimitta/(suurempien puiden aritmeettinen keskilapimitta)
            SUMD=0.
            NISO=0
            DO 18 K=1,NE
            IF(ED(K).GT.CD(I))THEN
               SUMD=SUMD+ED(K)
               NISO=NISO+1
            END IF
18          CONTINUE
            IF(NISO.GT.0)THEN
               E=CD(I)/(SUMD/REAL(NISO))
            ELSE
               E=1.
            END IF
	    CALL F_AGE1(AGE,V1,V2,CS(I),CD(I),E,TYP,TEM,DRA,DIS)
	  ELSE
	    CALL F_AGE2(AGE,V1,V2,CS(I),CD(I),E,TYP,TEM,DRA,DIS)
	  ENDIF
*	  ... ja laske metsikkotekijan ennuste
          ERO=ERO+(ALOG(CA(I))-ALOG(AGE))
12        CONTINUE
            IF(V1.NE.0)C=NC*V1/(V2+NC*V1)*(ERO/REAL(NC))
*	  ja talleta mallilaji ja kalibrointitiedot seuraavaa kutsua
*         varten
	  LAJIE=LAJI
	  DO 5 I=1,NC
	    CSE(I)=CS(I)
	    CDE(I)=CD(I)
5	  CAE(I)=CA(I)
          TYPE=TYP
          TEME=TEM
          DRAE=DRA
          DISE=DIS
3	  CONTINUE
*       puulajien valiset erot kalibr
        CLAJI=CLA(F,S,CS(1),C,DRA)
*	MUUTEN ELI JOS EI OLE KALIBROINTITUNNUKSIA ...
	ELSE
*	  ... METSIKKOTEKIJA = 0
	  C=0
	ENDIF

*	LASKE TARKASTELTAVAN PUUN IKA MALLEILLA
*	---------------------------------------

	IF(LAJI.EQ.1)THEN
*	  laske puun suhteellinen koko
*	    suhteellinen koko = 
*	    lapimitta/(suurempien puiden aritmeettinen keskilapimitta)
            SUMD=0.
            NISO=0
            DO 28 K=1,NE
            IF(ED(K).GT.D)THEN
               SUMD=SUMD+ED(K)
               NISO=NISO+1
            END IF
28          CONTINUE
            IF(NISO.GT.0)THEN
               E=D/(SUMD/REAL(NISO))
            ELSE
               E=1
            END IF
	    CALL F_AGE1(AGE,V1,V2,S,D,E,TYP,TEM,DRA,DIS)
	ELSE
	  CALL F_AGE2(AGE,V1,V2,S,D,E,TYP,TEM,DRA,DIS)
	ENDIF


*	KORJAA MALLIEN TULOS KALIBROINTIKERTOIMELLA
*	-------------------------------------------

*        IF(C.EQ.0.OR.F.GT.4) THEN
        IF(C.EQ.0) THEN
	  F_AGE=AGE*EXP((V1+V2)/2.)
	  IND_C=0
        ELSE        
	  F_AGE=AGE*EXP(CLAJI)*exp(v2/2)
	  IND_C=1
        END IF

*	testitulostus
*	write(6,'(1h ,2i5,10f10.3)')nc,ind_c,c,e,s,d,age,f_age

	RETURN

	END

*       metsikkotekijan muunnos eri puulajeille
        FUNCTION CLA(F,S,CS,C,DRA)
*       x(kangas/suo,laji,kalibr.puun laji,2)
        REAL X(4,5,5,2),Y(2,8,2)
        INTEGER LAJI(2,8),IAR(6)
*       MANTY
        DATA (((X(I,1,J,K),K=1,2),J=1,5),I=1,4)/
* KANGAS
     -  .00,1.00,.09,.65,.01,.58,-.04,.48,0.03,0.,
* SUO (LUONN.til)
     -  .00,1.00,-.08,.00,.00,.00,-.05,.00,.0,.0,
* SUO (MUUTTUMA)
     -  .00,1.00,.08,.36,.00,.00,-.05,.35,.0,.0,
* SUO (turvekangas))
     -  .00,1.00,.09,.61,.00,.00,-.08,.34,.0,.0/
*       KUUSI
        DATA (((X(I,2,J,K),K=1,2),J=1,5),I=1,4)/
     -  -.09,.81,.00,1.00,-.12,.42,-.14,.55,-.07,.0,
     -  -.07,.00,.00,1.00,.00,.00,-.19,.51,.0,.0,
     -  -.05,.57,.00,1.00,.00,.00,-.25,.55,.0,.0,
     -  -.10,.76,.00,1.00,.00,.00,-.09,.41,.0,.0/
*       RAUDUS
        DATA ((X(1,3,J,K),K=1,2),J=1,5)/
     -  -.01,.54,.07,.44,.00,1.00,-.02,.58,.0,.0/
*       HIES
        DATA (((X(I,4,J,K),K=1,2),J=1,5),I=1,4)/
     -  .04,.62,.14,.58,-.02,.58,.0,1.,.04,.49,
     -  .16,.00,.21,.75,.0,.0,.0,1.,.0,.0,
     -  .13,.38,.26,.48,.0,.0,.0,1.,.0,.0,
     -  .12,.46,.16,.64,.0,.0,.0,1.,.0,.0/
*       HAAPA
        DATA ((X(1,5,J,K),K=1,2),J=1,5)/
     -  .14,.0,.07,.0,.0,.0,-.05,.7,.0,1.0/
*       LATVUSKERROKSEN VAIKUTUS
*       MANTY
        DATA ((Y(1,J,K),K=1,2),J=1,8)/
     -  .0,1.,.0,1.,.0,1.,.19,.98,.00,.00,
     -  .0,1.,.0,1.,.00,1.00/
*       KUUSI
        DATA ((Y(2,J,K),K=1,2),J=1,8)/
     -  -.06,.69,.0,1.,.0,1.,.09,.95,.13,.42,
     -   .0,1.,.0,1.,.13,.42/
* kankaat
      data (laji(1,k),k=1,8)/1,2,3,4,5,4,1,4/
* suot
      data (laji(2,k),k=1,8)/1,2,4,4,4,4,1,4/
*     alaryhma ojitustilanteesta
      data iar/1,1,2,2,3,4/

        IF(S.EQ.CS)THEN
           CLA1=C
        ELSE
           IA=IAR(IFIX(DRA)+1)
           IAP=MIN0(IA,2)
           LAJIS=LAJI(IAP,IFIX(S))
           LAJICS=LAJI(IAP,IFIX(CS))
           CLA1=X(IA,LAJIS,LAJICS,1)+
     -         X(IA,LAJIS,LAJICS,2)*C
        END IF
* LATVUSKERROKSET MANNYLLA JA KUUSELLA
        IF(S.LE.2.AND.(F.EQ.1.OR.F.GE.4))THEN
                 CLA=CLA1*Y(IFIX(S),IFIX(F),2)+
     -              Y(IFIX(S),IFIX(F),1)
        IF(S.EQ.1.AND.F.EQ.8)CLA=AMAX1(.05,CLA)
        ELSE
           CLA=CLA1
        END IF
        RETURN
        END

*	======
*	MALLIT
*	======

*	- puuttuu - TARKEMPI MALLIEN, TOIMINNAN JA KAYTON SELITYS

*	-----------------
	SUBROUTINE F_AGE1(AGE,V1,V2,S,D,E,TYP,TEM,DRA,DIS)
*	-----------------
*       S puulaji
*       D rinnankorkeuslapimitta
*	E puun suhteelline koko
*	  lapimitta/(suurempien puiden aritmeettinen keskilapimitta)
*       TYP kasvupaikkatyyppi (VMI 7)
*       TEM lamposumma (d.d)
*       DRA ojitustilanne
*       DIS etaisyys rannikosta

* IKAMALLIT (SUHT. KOKO selittajana)
* selitettava muuttuja log(rinnankorkeusika)

* kertoimet pa-taulukossa (alaryhma,tyyppiryhma,sisamaa/rannikko
* (<2 km mereen, kaytossa mannylla ja kuusella),puulaji) 
* pa(1)=vakio      
* pa(2)=log(d)
* pa(3)=suht.koko
* pa(4)=suht.koko**2.
* pa(5)=lamposumma/10.
* pa(7)= 1, jos muuttuma, 0 muuten
* pa(8)= 1, jos turvekangas, 0 muuten

* varianssikomponentit va-taulukossa (indeksit kuten pa)
* va(1)=koealojen valinen
* va(2)=koealojen sisainen

      dimension pa(2,5,2,5,8),v(2,5,2,5,2),laji(2,8),
     *iar(6),ity(5,8),taso(2,8)

c   MANNYN kertoimet KANGAS
c sisamaa
* tyypit 1 ja 2
      data (pa(1,1,1,1,k),k=1,8)
     & /.7941,.7937,.5937,-.8476,.005583,.0,.0,.0/
* tyyppi 3
      data (pa(1,2,1,1,k),k=1,8)
     & /2.408,1.032,-1.003,-.1578,-.005112,.0,.0,.0/
* tyyppi 4 
      data (pa(1,3,1,1,k),k=1,8)
     & /2.610,.9870,-.6908,-.2429,-.006360,.0,.0,.0/
* tyypit 5 ja 6
      data (pa(1,4,1,1,k),k=1,8)
     & /2.200,.9376,-.4663,-.3329,-.0009766,.0,.0,.0/
* tyyppi 7
      data (pa(1,5,1,1,k),k=1,8)
     & /3.105,.6365,.8950,-.7346,-.007221,.0,.0,.0/
* 1,2
      data (v(1,1,1,1,k),k=1,2)/.0989,.0189/
* 3
      data (v(1,2,1,1,k),k=1,2)/.0948,.0202/
* 4 
      data (v(1,3,1,1,k),k=1,2)/.1208,.0266/
* 5,6
      data (v(1,4,1,1,k),k=1,2)/.1041,.0270/
* 7
      data (v(1,5,1,1,k),k=1,2)/.1168,.0382/
c rannikko (MT perustaso)
      data (pa(1,1,2,1,k),k=1,8)/1.717,.7480,1.410,
     *-1.177,-.002251,.0,.0,.0/
      data (v(1,1,2,1,k),k=1,2)/.1263,.0343/
c muiden kasvup. tyyppien tasoerot rannikolla
* tyypit 1 ja 2
      data (taso(1,k),k=1,5)/-.2918,
* tyyppi 3
     *                        .0,
* tyyppi 4
     *                        -.1041,
* tyyppi 5
     *                        .7878,
* tyypit 6 ja 7
     *                        .3880/
                                
c KUUSEN kertoimet KANGAS
* tyypit 1 ja 2
      data (pa(1,1,1,2,k),k=1,8)
     & /2.238,.6208,.0001021,-.2177,-.001382,.0,.0,.0/
* tyyppi 3
      data (pa(1,2,1,2,k),k=1,8)
     & /3.949,.5861,-.07581,-.1568,-.01278,.0,.0,.0/
* tyypit 4,...
      data (pa(1,3,1,2,k),k=1,8)
     & /3.734,.8214,-.5546,-.1263,-.01297,.0,.0,.0/
c varianssikomponenttien est.
* 1,2
      data (v(1,1,1,2,k),k=1,2)/.0843,.0326/
* 3
      data (v(1,2,1,2,k),k=1,2)/.1025,.0328/
* 4,...
      data (v(1,3,1,2,k),k=1,2)/.1582,.0355/
c rannikko
      data (pa(1,1,2,2,k),k=1,8)/2.587,.6587,.1758,
     *-.2340,-.003961,.0,.0,.0/
ccc muiden kasvup. tyyppien tasoerot
* tyypit 1 ja 2
      data (taso(2,k),k=1,3)/-.3570,
* tyyppi 3
     *                        .0,
* tyyppi 4,...
     *                        .1380/
      data (v(1,1,2,2,k),k=1,2)/.1099,.0417/

c RAUDUSKOIVUN kertoimet KANGAS
* tyypit 1 ja 2
      data (pa(1,1,1,3,k),k=1,8)
     & /1.318,1.049,-1.816,.5385,.003332,.0,.0,.0/
* tyyppi 3
      data (pa(1,2,1,3,k),k=1,8)
     & /2.703,.8300,-.9823,-.1765,-.002619,.0,.0,.0/
* tyypit 4,...
      data (pa(1,3,1,3,k),k=1,8)
     & /1.595,1.012,3.474,-3.385,-.009258,.0,.0,.0/
* 1,2
      data (v(1,1,1,3,k),k=1,2)/.0653,.0421/
* 3
      data (v(1,2,1,3,k),k=1,2)/.0853,.0142/
* 4,...
      data (v(1,3,1,3,k),k=1,2)/.1068,.0229/

c HIESKOIVUN kertoimet KANGAS
* tyypit 1 ja 2
      data (pa(1,1,1,4,k),k=1,8)
     & /1.571,.8486,-1.331,.3197,.005055,.0,.0,.0/
* tyyppi 3
      data (pa(1,2,1,4,k),k=1,8)
     & /3.071,.8419,-1.747,.5970,-.005448,.0,.0,.0/
* tyypit 4,...
      data (pa(1,3,1,4,k),k=1,8)
     & /3.131,.7289,-.6106,-.1430,-.007071,.0,.0,.0/
* 1,2
      data (v(1,1,1,4,k),k=1,2)/.0810,.0391/
* 3
      data (v(1,2,1,4,k),k=1,2)/.1114,.0415/
* 4,...
      data (v(1,3,1,4,k),k=1,2)/.2006,.0380/

c HAAVAN kertoimet KANGAS
* kaikki kasvup.
      data (pa(1,1,1,5,k),k=1,8)
     & /2.140,.7451,-.1353,-.4921,-.002676,.0,.0,.0/
      data (v(1,1,1,5,k),k=1,2)/.0815,.0386/

C MANNYN KERTOIMET TURVEMAA
* tyypit 1 ja 2
      data (pa(2,1,1,1,k),k=1,8)
     & /2.552,.5609,.8413,-.5746,-.003796,.0,-.1826,-.1884/
* tyyppi 3
      data (pa(2,2,1,1,k),k=1,8)
     & /2.912,.7470,.1360,-.4414,-.006003,.0,-.2502,-.3167/
* tyyppi 4
      data (pa(2,3,1,1,k),k=1,8)
     & /2.887,.7966,-.2111,-.1218,-.004291,.0,-.2699,-.4083/
* tyypit 5 ja 6
      data (pa(2,4,1,1,k),k=1,8)
     & /2.412,.7744,1.287,-.9452,-.003994,.0,-.3402,-.3715/
      data (v(2,1,1,1,k),k=1,2)/.0839,.0433/
      data (v(2,2,1,1,k),k=1,2)/.1313,.0403/
      data (v(2,3,1,1,k),k=1,2)/.1489,.0591/
      data (v(2,4,1,1,k),k=1,2)/.1801,.0886/

c KUUSEN kertoimet TURVEMAA
* tyypit 1 ja 2
      data (pa(2,1,1,2,k),k=1,8)
     & /3.443,.4191,1.585,-.8903,-.01155,.0,.01959,-.1758/
* tyyppi 3
      data (pa(2,2,1,2,k),k=1,8)
     & /4.244,.5311,.5069,-.3456,-.01432,.0,-.1761,-.2843/
* tyypit 4,...
      data (pa(2,3,1,2,k),k=1,8)
     & /3.863,.7665,-.2844,-.1590,-.01137,.0,-.2315,-.3988/
* 1,2
      data (v(2,1,1,2,k),k=1,2)/.1023,.0594/
* 3
      data (v(2,2,1,2,k),k=1,2)/.0938,.0551/
* 4,...
      data (v(2,3,1,2,k),k=1,2)/.1533,.0811/

c HIESKOIVUN kertoimet TURVEMAA
* tyypit 1 ja 2
      data (pa(2,1,1,4,k),k=1,8)
     & /2.356,.6887,-.8660,.2717,-.0000804,.0,-.05375,-.2313/
* tyyppi 3
      data (pa(2,2,1,4,k),k=1,8)
     & /2.364,.7068,.1273,-.4736,.00007545,.0,-.3491,-.3760/
* tyyppi 4,...
      data (pa(2,3,1,4,k),k=1,8)
     & /3.005,.7601,.1829,-.5385,-.006270,.0,-.4402,-.4872/
* 1,2
      data (v(2,1,1,4,k),k=1,2)/.1042,.0417/
* 3
      data (v(2,2,1,4,k),k=1,2)/.1252,.0538/
* 4,..
      data (v(2,3,1,4,k),k=1,2)/.1671,.0554/

*     puulajit
* kankaat
      data (laji(1,k),k=1,8)/1,2,3,4,5,5,1,5/
* suot
      data (laji(2,k),k=1,8)/1,2,4,4,4,4,1,4/
*     kasvupaikat
*     manty
      data (ity(1,k),k=1,8)/1,1,2,3,4,4,5,5/
*     kuusi
      data (ity(2,k),k=1,8)/1,1,2,3,3,3,3,3/
*     raudus
      data (ity(3,k),k=1,8)/1,1,2,3,3,3,3,3/
*     hies
      data (ity(4,k),k=1,8)/1,1,2,3,3,3,3,3/
*     haapa
      data (ity(5,k),k=1,8)/1,1,1,1,1,1,1,1/
*     alaryhma ojitustilanteesta
      data iar/1,1,2,2,2,2/

* alaryhma
        IA=IAR(IFIX(DRA)+1)
* puulaji
        IS=LAJI(IA,IFIX(S))
* tyyppiryhma
        ITYP=ITY(IS,IFIX(TYP))
      
*	ETAISYYDEN RANNIKOSTA OLETUSARVO
	DISE=DIS
	IF(DISE.LE.0)DISE=2.5
*       rannikolla eri mallit vain kankaiden kuusella ja mannylla
        IF(DISE.GT.2.OR.IA.EQ.2.OR.LAJI(IA,IFIX(S))
     -     .GT.2)THEN
          IDIS=1
        ELSE
          IDIS=2
*         rannikolla ei eri kasvup. omia malleja
          ITYP=1
        END IF

* vakio
        AGE=PA(IA,ITYP,IDIS,IS,1)
     -     +PA(IA,ITYP,IDIS,IS,2)*ALOG(D)
     -     +PA(IA,ITYP,IDIS,IS,3)*E
     -     +PA(IA,ITYP,IDIS,IS,4)*E**2.
     -     +PA(IA,ITYP,IDIS,IS,5)*TEM/10.

*       muuttumat
        IF(DRA.EQ.4)AGE=AGE+PA(IA,ITYP,IDIS,IS,7)
*       turvekankaat
        IF(DRA.EQ.5)AGE=AGE+PA(IA,ITYP,IDIS,IS,8)
*       rannikon kasvup. tasokorj.
        IF(IDIS.EQ.2)AGE=AGE+TASO(IS,ITY(IS,IFIX(TYP)))

        AGE=EXP(AGE)

*       varianssikomponentit
        V1=V(IA,ITYP,IDIS,IS,1)
        V2=V(IA,ITYP,IDIS,IS,2)

	RETURN

	END


*	- puuttuu - TARKEMPI MALLIEN, TOIMINNAN JA KAYTON SELITYS

*	---------------
	SUBROUTINE F_AGE2(AGE,V1,V2,S,D,E,TYP,TEM,DRA,DIS)
*	---------------

C KANKAIDEN IKAMALLIT
c selitettava muuttuja log(rinnankorkeusika)

c kertoimet pa-taulukossa (malli,tyyppiryhma,sisamaa/rannikko
* (<2 km mereen, kaytossa mannylla ja kuusella),puulaji)
c pa(1)=vakio      
c pa(2)=log(d)
c pa(3)=suht.koko
c pa(4)=suht.koko**2.
c pa(5)=lamposumma/10.
c pa(7)= 1, jos muuttuma, 0 muuten
c pa(8)= 1, jos turvekangas, 0 muuten

c varianssikomponentit va-taulukossa (indeksit kuten pa)
c va(1)=koealojen valinen
c va(2)=koealojen sisainen

      dimension pa(2,5,2,5,8),v(2,5,2,5,2),laji(2,8),
     *iar(6),ity(5,8),taso(2,8)

c   MANNYN kertoimet KANGAS
* tyypit 1 ja 2
      data (pa(1,1,1,1,k),k=1,8)
     & /1.410,.5204,.0,.0,.006734,.0,.0,.0/
* tyyppi 3
      data (pa(1,2,1,1,k),k=1,8)
     & /2.856,.5098,.0,.0,-.003644,.0,.0,.0/
* tyyppi 4 
      data (pa(1,3,1,1,k),k=1,8)
     & /2.994,.5407,.0,.0,-.005251,.0,.0,.0/
* tyypit 5 ja 6
      data (pa(1,4,1,1,k),k=1,8)
     & /2.318,.5432,.0,.0,-.001768,.0,.0,.0/
* tyyppi 7
      data (pa(1,5,1,1,k),k=1,8)
     & /3.392,.5663,.0,.0,-.006168,.0,.0,.0/
* 1,2
      data (v(1,1,1,1,k),k=1,2)/.1090,.0201/
* 3
      data (v(1,2,1,1,k),k=1,2)/.1301,.0213/
* 4 
      data (v(1,3,1,1,k),k=1,2)/.1704,.0277/
* 5,6
      data (v(1,4,1,1,k),k=1,2)/.1285,.0287/
* 7
      data (v(1,5,1,1,k),k=1,2)/.1196,.0393/
c rannikko (MT perustaso)
      data (pa(1,1,2,1,k),k=1,8)
     & /2.324,.6148,.0,.0,-.001067,.0,.0,.0/
      data (v(1,1,2,1,k),k=1,2)/.1263,.0343/
ccc muiden kasvup. tyyppien tasoerot
* tyypit 1 ja 2
      data (taso(1,k),k=1,5)/-.2633,
* tyyppi 3
     *                        .0,
* tyyppi 4
     *                       -.1520,
* tyyppi 5
     *                        .7880,
* tyypit 6 ja 7     
     *                        .3420/

c KUUSEN kertoimet KANGAS
* tyypit 1 ja 2
      data (pa(1,1,1,2,k),k=1,8)
     & /2.463,.5121,.0,.0,-.001699,.0,.0,.0/
* tyyppi 3
      data (pa(1,2,1,2,k),k=1,8)
     & /4.102,.4779,.0,.0,-.01272,.0,.0,.0/
* tyypit 4,...
      data (pa(1,3,1,2,k),k=1,8)
     & /4.100,.5642,.0,.0,-.01422,.0,.0,.0/
c varianssikomponenttien est.
* 1,2
      data (v(1,1,1,2,k),k=1,2)/.0898,.0330/
* 3
      data (v(1,2,1,2,k),k=1,2)/.1063,.0330/
* 4,...
      data (v(1,3,1,2,k),k=1,2)/.1911,.0362/
c rannikko
      data (pa(1,1,2,2,k),k=1,8)
     & /2.666,.6073,.0,.0,-.003463,.0,.0,.0/
ccc muiden kasvup. tyyppien tasoerot
* tyypit 1 ja 2
      data (taso(2,k),k=1,3)/-.3685,
* tyyppi 3
     *                        .0,
* tyyppi 4,...
     *                        .1336/
      data (v(1,1,2,2,k),k=1,2)/.1098,.0425/


c RAUDUSKOIVUN kertoimet KANGAS
* tyypit 1 ja 2
      data (pa(1,1,1,3,k),k=1,8)
     & /1.226,.6637,.0,.0,.004359,.0,.0,.0/
* tyyppi 3
      data (pa(1,2,1,3,k),k=1,8)
     & /3.286,.3385,.0,.0,-.003205,.0,.0,.0/
* tyypit 4,...
      data (pa(1,3,1,3,k),k=1,8)
     & /2.962,.8869,.0,.0,-.01293,.0,.0,.0/
* 1,2
      data (v(1,1,1,3,k),k=1,2)/.0834,.0426/
* 3
      data (v(1,2,1,3,k),k=1,2)/.1062,.0143/
* 4,...
      data (v(1,3,1,3,k),k=1,2)/.0931,.0534/

c HIESKOIVUN kertoimet KANGAS
* tyypit 1 ja 2
      data (pa(1,1,1,4,k),k=1,8)
     & /1.270,.5243,.0,.0,.008244,.0,.0,.0/
* tyyppi 3
      data (pa(1,2,1,4,k),k=1,8)
     & /2.800,.4818,.0,.0,-.002802,.0,.0,.0/
* tyypit 4,...
      data (pa(1,3,1,4,k),k=1,8)
     & /2.935,.4916,.0,.0,-.004875,.0,.0,.0/
* 1,2
      data (v(1,1,1,4,k),k=1,2)/.1210,.0393/
* 3
      data (v(1,2,1,4,k),k=1,2)/.1654,.0414/
* 4,...
      data (v(1,3,1,4,k),k=1,2)/.2413,.0389/

c HAAVAN kertoimet KANGAS
* kaikki kasvup.
      data (pa(1,1,1,5,k),k=1,8)
     & /2.398,.4876,.0,.0,-.002298,.0,.0,.0/
      data (v(1,1,1,5,k),k=1,2)/.1113,.0405/


C TURVEMAIDEN IKAMALLIT

c selitettava muuttuja log(rinnankorkeusika)
c kertoimet pa-taulukossa (malli,tyyppiryhma,sisamaa/rannikko(suolla ei
c                          kaytossa),puulaji) 
c pa(1)=vakio      
c pa(2)=log(d)
c pa(3)=suht.koko
c pa(4)=suht.koko**2.
c pa(5)=lamposumma/10.
c pa(7)= 1, jos muuttuma, 0 muuten
c pa(8)= 1, jos turvekangas, 0 muuten

c varianssikomponentit va-taulukossa (indeksit kuten pa)
c va(1)=koealojen valinen
c va(2)=koealojen sisainen

C MANNYN KERTOIMET TURVEMAA
* tyypit 1 ja 2
      data (pa(2,1,1,1,k),k=1,8)
     & /2.816,.5649,.0,.0,-.003687,.0,-.1813,-.1885/
* tyyppi 3
      data (pa(2,2,1,1,k),k=1,8)
     & /2.986,.5467,.0,.0,-.003639,.0,-.2396,-.3047/
* tyyppi 4
      data (pa(2,3,1,1,k),k=1,8)
     & /2.954,.6378,.0,.0,-.003719,.0,-.2722,-.3701/
* tyypit 5 ja 6
      data (pa(2,4,1,1,k),k=1,8)
     & /2.771,.7734,.0,.0,-.003792,.0,-.3383,-.3614/
      data (v(2,1,1,1,k),k=1,2)/.0825,.0450/
      data (v(2,2,1,1,k),k=1,2)/.1398,.0415/
      data (v(2,3,1,1,k),k=1,2)/.1556,.0598/
      data (v(2,4,1,1,k),k=1,2)/.1788,.0921/

c KUUSEN kertoimet TURVEMAA
* tyypit 1 ja 2
      data (pa(2,1,1,2,k),k=1,8)
     & /3.707,.6038,.0,.0,-.01265,.0,.02898,-.1818/
* tyyppi 3
      data (pa(2,2,1,2,k),k=1,8)
     & /4.374,.5560,.0,.0,-.01452,.0,-.1760,-.2854/
* tyyppi 4,...
      data (pa(2,3,1,2,k),k=1,8)
     & /3.771,.5743,.0,.0,-.00875,.0,-.2948,-.4633/
* 1,2
      data (v(2,1,1,2,k),k=1,2)/.1480,.0588/
* 3
      data (v(2,2,1,2,k),k=1,2)/.0935,.0553/
* 4,...
      data (v(2,3,1,2,k),k=1,2)/.1651,.0804/

c HIESKOIVUN kertoimet TURVEMAA
* tyypit 1 ja 2
      data (pa(2,1,1,4,k),k=1,8)
     & /2.088,.5001,.0,.0,.002017,.0,-.05694,-.2257/
* tyyppi 3
      data (pa(2,2,1,4,k),k=1,8)
     & /2.361,.5628,.0,.0,.001504,.0,-.3358,-.3452/
* tyyppi 4,...
      data (pa(2,3,1,4,k),k=1,8)
     & /3.115,.6270,.0,.0,-.006271,.0,-.4392,-.4598/
* 1,2
      data (v(2,1,1,4,k),k=1,2)/.1307,.0423/
* 3
      data (v(2,2,1,4,k),k=1,2)/.1394,.0548/
* 4,..
      data (v(2,3,1,4,k),k=1,2)/.2019,.0575/

*     puulajit
* kankaat
      data (laji(1,k),k=1,8)/1,2,3,4,5,5,1,5/
* suot
      data (laji(2,k),k=1,8)/1,2,4,4,4,4,1,4/
*     kasvupaikat
*     manty
      data (ity(1,k),k=1,8)/1,1,2,3,4,4,5,5/
*     kuusi
      data (ity(2,k),k=1,8)/1,1,2,3,3,3,3,3/
*     raudus
      data (ity(3,k),k=1,8)/1,1,2,3,3,3,3,3/
*     hies
      data (ity(4,k),k=1,8)/1,1,2,3,3,3,3,3/
*     haapa
      data (ity(5,k),k=1,8)/1,1,1,1,1,1,1,1/
*     alaryhma ojitustilanteesta
      data iar/1,1,2,2,2,2/

* alaryhma
        IA=IAR(IFIX(DRA)+1)
* puulaji
        IS=LAJI(IA,IFIX(S))
* tyyppiryhma
        ITYP=ITY(IS,IFIX(TYP))
      
*	ETAISYYDEN RANNIKOSTA OLETUSARVO
	DISE=DIS
	IF(DISE.LE.0)DISE=2.5
*       rannikolla eri mallit vain kankaiden kuusella ja mannylla
        IF(DISE.GT.2.OR.IA.EQ.2.OR.LAJI(IA,IFIX(S))
     -     .GT.2)THEN
          IDIS=1
        ELSE
          IDIS=2
*         rannikolla ei eri kasvup. omia malleja
          ITYP=1
        END IF

* vakio
        AGE=PA(IA,ITYP,IDIS,IS,1)
     -     +PA(IA,ITYP,IDIS,IS,2)*ALOG(D)
     -     +PA(IA,ITYP,IDIS,IS,5)*TEM/10.
 
*       muuttumat
        IF(DRA.EQ.4)AGE=AGE+PA(IA,ITYP,IDIS,IS,7)
*       turvekankaat
        IF(DRA.EQ.5)AGE=AGE+PA(IA,ITYP,IDIS,IS,8)
*       rannikon kasvup. tasokorj.
        IF(IDIS.EQ.2)AGE=AGE+TASO(IS,ITY(IS,IFIX(TYP)))

        AGE=EXP(AGE)

*       varianssikomponentit
        V1=V(IA,ITYP,IDIS,IS,1)
        V2=V(IA,ITYP,IDIS,IS,2)

	RETURN

	END
      
