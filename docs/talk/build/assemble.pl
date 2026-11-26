use strict; use warnings; use JSON::PP;
my $J = JSON::PP->new->utf8->canonical;
sub slurp { local $/; open my $h, '<', shift or die; <$h> }
my $bld = $J->decode(slurp('psych_build.json'));
my @r = @{ $bld->{requests} };
sub rgb { my $h = shift; $h =~ s/#//; my @c = map { hex($_)/255 } ($h =~ /(..)(..)(..)/); { red=>$c[0], green=>$c[1], blue=>$c[2] } }
# bold first column of new tables (rows after header)
my %rows = (ps1_tbl=>5, ps2_tbl=>7, ps4_tbl=>3);
for my $t (sort keys %rows) { for my $i (1..$rows{$t}) {
  push @r, { updateTextStyle => { objectId=>$t, cellLocation=>{rowIndex=>$i, columnIndex=>0}, textRange=>{type=>'ALL'}, style=>{bold=>JSON::PP::true}, fields=>'bold' } } } }
# tint the TRELLIS row of the segmentation table
push @r, { updateTableCellProperties => { objectId=>'ps2_tbl', tableRange=>{location=>{rowIndex=>1, columnIndex=>0}, rowSpan=>1, columnSpan=>4},
  tableCellProperties=>{tableCellBackgroundFill=>{solidFill=>{color=>{rgbColor=>rgb('#E6EEF3')}}}}, fields=>'tableCellBackgroundFill' } };
# digit-span figure, 1440x880 -> 5.4 x 3.3 in
my $E = 914400;
push @r, { createImage => { objectId=>'ps3_img', url=>'https://drive.google.com/uc?export=download&id=1eObh5UrT4b3YzftlmYmirl514_uTwmjo',
  elementProperties=>{ pageObjectId=>'ps3_slide', size=>{width=>{magnitude=>int(5.4*$E), unit=>'EMU'}, height=>{magnitude=>int(3.3*$E), unit=>'EMU'}},
  transform=>{scaleX=>1, scaleY=>1, shearX=>0, shearY=>0, translateX=>int(0.5*$E), translateY=>int(1.0*$E), unit=>'EMU'} } } };
# renumber: old 8..28 -> +1, old 29..44 -> +5
my @num = qw(p1_post_num p1_steps_num p1_info_num t07_num p2_post_num p2_steps_num p2_info_num t08_num l1_steps_num l1_info_num t10_num t15_num f03_num f01_num t20_num f06_num f08_num f10_num SLIDES_API371542853_0 t23_num c30_num
             t24_num t25_num t26_num t05_num p1_win_num f02_num t18_num t19_num t13_num f07_num f09_num f04_num t14_num f11_num t27_num t29_num);
my $old = 8;
for my $id (@num) { my $new = $old <= 28 ? $old + 1 : $old + 5;
  push @r, { deleteText => { objectId=>$id, textRange=>{type=>'ALL'} } }, { insertText => { objectId=>$id, insertionIndex=>0, text=>"$new" } }; $old++ }
# references: insert in descending index order
my $refs = slurp('txt_t26_bul.txt');
my @ins = (
 ['Gobet, F.', "Frank, M. C., Goldwater, S., Griffiths, T. L., & Tenenbaum, J. B. (2010). Modeling human performance in statistical word segmentation. Cognition, 117, 107-125.\n"],
 ['Langley, P.', "Knowlton, B. J., & Squire, L. R. (1996). Artificial grammar learning depends on implicit acquisition of both abstract and exemplar-specific information. JEP: LMC, 22, 169-181.\n"],
 ['Miller, G.', "Meulemans, T., & Van der Linden, M. (1997). Associative chunk strength in artificial grammar learning. JEP: LMC, 23, 1007-1028.\n"],
 ['Singaravadivelan, K.', "Richman, H. B., Staszewski, J. J., & Simon, H. A. (1995). Simulation of expert memory using EPAM IV. Psychological Review, 102, 305-330.\nSaffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. Science, 274, 1926-1928.\n"]);
my @pos = map { my $i = index($refs, $_->[0]); die "missing $_->[0]" if $i < 0; [$i, $_->[1]] } @ins;
for my $p (sort { $b->[0] <=> $a->[0] } @pos) { push @r, { insertText => { objectId=>'t26_bul', insertionIndex=>$p->[0], text=>$p->[1] } } }
# future work: one more open question, and move the two boxes up 0.3 in
my $fut = slurp('txt_t24_bul.txt'); my $fl = length($fut) - 1;
push @r, { insertText => { objectId=>'t24_bul', insertionIndex=>$fl, text=>"\nCan a limit on memory recover the psychological effects that TRELLIS misses?" } };
push @r, { updatePageElementTransform => { objectId=>'t24_qh', applyMode=>'ABSOLUTE', transform=>{scaleX=>2.7432, scaleY=>0.1219, shearX=>0, shearY=>0, translateX=>457200, translateY=>3794760-274320, unit=>'EMU'} } };
push @r, { updatePageElementTransform => { objectId=>'t24_bul', applyMode=>'ABSOLUTE', transform=>{scaleX=>2.7432, scaleY=>0.3048, shearX=>0, shearY=>0, translateX=>457200, translateY=>4069080-274320, unit=>'EMU'} } };
# conclusion: append a bullet, then replace the first
my $con = slurp('txt_t25_bul.txt'); my $cl = length($con) - 1;
push @r, { insertText => { objectId=>'t25_bul', insertionIndex=>$cl, text=>"\nIn five psychological studies the chunks reproduce familiarity effects; the misses need a limit on memory." } };
push @r, { replaceAllText => { containsText=>{text=>'Version 2 refines three postulates and keeps the rest.', matchCase=>JSON::PP::true}, replaceText=>'Version 2 keeps the postulates of version 1 and adds six addenda.', pageObjectIds=>['t25_slide'] } };
# title slide
push @r, { replaceAllText => { containsText=>{text=>'TRELLIS v2: Chunking as Compression', matchCase=>JSON::PP::true}, replaceText=>'TRELLIS: Unsupervised Learning of Concepts and Chunks from Raw Experience', pageObjectIds=>['t01_slide'] } };
push @r, { replaceAllText => { containsText=>{text=>'Learning concepts and chunks from raw text and chess, one experience at a time', matchCase=>JSON::PP::true}, replaceText=>'Chess, children\'s stories and five psychological experiments, learned one experience at a time', pageObjectIds=>['t01_slide'] } };
push @r, { updatePageElementTransform => { objectId=>'t01_title', applyMode=>'ABSOLUTE', transform=>{scaleX=>2.7432, scaleY=>0.5639, shearX=>0, shearY=>0, translateX=>457200, translateY=>365760, unit=>'EMU'} } };
print $J->encode({ requests=>\@r, writeControl=>$bld->{writeControl} });
