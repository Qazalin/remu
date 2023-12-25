
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_4_4_5_13_3_3_3>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[8:9], s[0:1], 0x10                            // 00000000170C: F4040200 F8000010
	s_mul_hi_i32 s1, s13, 0x4ec4ec4f                           // 000000001714: 9701FF0D 4EC4EC4F
	s_mul_i32 s0, s15, 0x39c                                   // 00000000171C: 9600FF0F 0000039C
	s_lshr_b32 s2, s1, 31                                      // 000000001724: 85029F01
	s_ashr_i32 s3, s1, 2                                       // 000000001728: 86038201
	s_ashr_i32 s1, s0, 31                                      // 00000000172C: 86019F00
	s_add_i32 s3, s3, s2                                       // 000000001730: 81030203
	s_lshl_b64 s[10:11], s[0:1], 2                             // 000000001734: 848A8200
	s_mul_i32 s2, s3, 13                                       // 000000001738: 96028D03
	s_mul_i32 s12, s3, 56                                      // 00000000173C: 960CB803
	s_sub_i32 s0, s13, s2                                      // 000000001740: 8180020D
	s_waitcnt lgkmcnt(0)                                       // 000000001744: BF89FC07
	s_add_u32 s1, s6, s10                                      // 000000001748: 80010A06
	s_addc_u32 s3, s7, s11                                     // 00000000174C: 82030B07
	s_ashr_i32 s13, s12, 31                                    // 000000001750: 860D9F0C
	s_mul_i32 s10, s14, 27                                     // 000000001754: 960A9B0E
	s_lshl_b64 s[6:7], s[12:13], 2                             // 000000001758: 8486820C
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_2) | instid1(SALU_CYCLE_1)// 00000000175C: BF8704B9
	s_add_u32 s1, s1, s6                                       // 000000001760: 80010601
	s_addc_u32 s3, s3, s7                                      // 000000001764: 82030703
	s_lshl_b32 s6, s0, 1                                       // 000000001768: 84068100
	s_ashr_i32 s7, s6, 31                                      // 00000000176C: 86079F06
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001770: BF870499
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001774: 84868206
	s_add_u32 s6, s1, s6                                       // 000000001778: 80060601
	s_addc_u32 s7, s3, s7                                      // 00000000177C: 82070703
	s_ashr_i32 s11, s10, 31                                    // 000000001780: 860B9F0A
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001784: BF870499
	s_lshl_b64 s[10:11], s[10:11], 2                           // 000000001788: 848A820A
	s_add_u32 s8, s8, s10                                      // 00000000178C: 80080A08
	s_addc_u32 s9, s9, s11                                     // 000000001790: 82090B09
	s_load_b512 s[16:31], s[8:9], null                         // 000000001794: F4100404 F8000000
	s_clause 0x3                                               // 00000000179C: BF850003
	s_load_b64 s[10:11], s[6:7], null                          // 0000000017A0: F4040283 F8000000
	s_load_b32 s1, s[6:7], 0x8                                 // 0000000017A8: F4000043 F8000008
	s_load_b64 s[12:13], s[6:7], 0x70                          // 0000000017B0: F4040303 F8000070
	s_load_b32 s3, s[6:7], 0x78                                // 0000000017B8: F40000C3 F8000078
	s_waitcnt lgkmcnt(0)                                       // 0000000017C0: BF89FC07
	v_fma_f32 v0, s10, s16, 0                                  // 0000000017C4: D6130000 0200200A
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 0000000017CC: BF8700C1
	v_fmac_f32_e64 v0, s11, s17                                // 0000000017D0: D52B0000 0000220B
	s_clause 0x1                                               // 0000000017D8: BF850001
	s_load_b64 s[10:11], s[6:7], 0x4d0                         // 0000000017DC: F4040283 F80004D0
	s_load_b64 s[16:17], s[6:7], 0xe0                          // 0000000017E4: F4040403 F80000E0
	v_fmac_f32_e64 v0, s1, s18                                 // 0000000017EC: D52B0000 00002401
	s_load_b32 s1, s[6:7], 0xe8                                // 0000000017F4: F4000043 F80000E8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017FC: BF870091
	v_fmac_f32_e64 v0, s12, s19                                // 000000001800: D52B0000 0000260C
	v_fmac_f32_e64 v0, s13, s20                                // 000000001808: D52B0000 0000280D
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001810: BF870001
	v_fmac_f32_e64 v0, s3, s21                                 // 000000001814: D52B0000 00002A03
	s_clause 0x2                                               // 00000000181C: BF850002
	s_load_b32 s3, s[6:7], 0x4d8                               // 000000001820: F40000C3 F80004D8
	s_load_b64 s[12:13], s[6:7], 0x5b0                         // 000000001828: F4040303 F80005B0
	s_load_b64 s[34:35], s[6:7], 0x540                         // 000000001830: F4040883 F8000540
	s_waitcnt lgkmcnt(0)                                       // 000000001838: BF89FC07
	v_fmac_f32_e64 v0, s16, s22                                // 00000000183C: D52B0000 00002C10
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001844: BF8700A1
	v_fmac_f32_e64 v0, s17, s23                                // 000000001848: D52B0000 00002E11
	s_load_b256 s[16:23], s[8:9], 0x40                         // 000000001850: F40C0404 F8000040
	v_fmac_f32_e64 v0, s1, s24                                 // 000000001858: D52B0000 00003001
	s_load_b32 s1, s[6:7], 0x548                               // 000000001860: F4000043 F8000548
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001868: BF870091
	v_fmac_f32_e64 v0, s10, s25                                // 00000000186C: D52B0000 0000320A
	v_fmac_f32_e64 v0, s11, s26                                // 000000001874: D52B0000 0000340B
	s_delay_alu instid0(VALU_DEP_1)                            // 00000000187C: BF870001
	v_fmac_f32_e64 v0, s3, s27                                 // 000000001880: D52B0000 00003603
	s_clause 0x2                                               // 000000001888: BF850002
	s_load_b32 s3, s[6:7], 0x5b8                               // 00000000188C: F40000C3 F80005B8
	s_load_b64 s[10:11], s[6:7], 0xa10                         // 000000001894: F4040283 F8000A10
	s_load_b64 s[24:25], s[6:7], 0x9a0                         // 00000000189C: F4040603 F80009A0
	v_fmac_f32_e64 v0, s34, s28                                // 0000000018A4: D52B0000 00003822
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000018AC: BF8700A1
	v_fmac_f32_e64 v0, s35, s29                                // 0000000018B0: D52B0000 00003A23
	s_waitcnt lgkmcnt(0)                                       // 0000000018B8: BF89FC07
	v_fmac_f32_e64 v0, s1, s30                                 // 0000000018BC: D52B0000 00003C01
	s_load_b32 s1, s[6:7], 0x9a8                               // 0000000018C4: F4000043 F80009A8
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000018CC: BF870091
	v_fmac_f32_e64 v0, s12, s31                                // 0000000018D0: D52B0000 00003E0C
	v_fmac_f32_e64 v0, s13, s16                                // 0000000018D8: D52B0000 0000200D
	s_delay_alu instid0(VALU_DEP_1)                            // 0000000018E0: BF870001
	v_fmac_f32_e64 v0, s3, s17                                 // 0000000018E4: D52B0000 00002203
	s_clause 0x1                                               // 0000000018EC: BF850001
	s_load_b32 s3, s[6:7], 0xa18                               // 0000000018F0: F40000C3 F8000A18
	s_load_b64 s[12:13], s[6:7], 0xa80                         // 0000000018F8: F4040303 F8000A80
	s_load_b64 s[16:17], s[8:9], 0x60                          // 000000001900: F4040404 F8000060
	v_fmac_f32_e64 v0, s24, s18                                // 000000001908: D52B0000 00002418
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001910: BF8700A1
	v_fmac_f32_e64 v0, s25, s19                                // 000000001914: D52B0000 00002619
	s_waitcnt lgkmcnt(0)                                       // 00000000191C: BF89FC07
	v_fmac_f32_e64 v0, s1, s20                                 // 000000001920: D52B0000 00002801
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001928: BF870001
	v_fmac_f32_e64 v0, s10, s21                                // 00000000192C: D52B0000 00002A0A
	s_load_b32 s1, s[6:7], 0xa88                               // 000000001934: F4000043 F8000A88
	s_load_b32 s10, s[8:9], 0x68                               // 00000000193C: F4000284 F8000068
	s_mul_i32 s6, s15, 0x104                                   // 000000001944: 9606FF0F 00000104
	s_mul_i32 s8, s14, 0x41                                    // 00000000194C: 9608FF0E 00000041
	s_ashr_i32 s7, s6, 31                                      // 000000001954: 86079F06
	v_fmac_f32_e64 v0, s11, s22                                // 000000001958: D52B0000 00002C0B
	s_lshl_b64 s[6:7], s[6:7], 2                               // 000000001960: 84868206
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_1)// 000000001964: BF8700C1
	v_fmac_f32_e64 v0, s3, s23                                 // 000000001968: D52B0000 00002E03
	s_add_u32 s3, s4, s6                                       // 000000001970: 80030604
	s_addc_u32 s6, s5, s7                                      // 000000001974: 82060705
	s_ashr_i32 s9, s8, 31                                      // 000000001978: 86099F08
	v_fmac_f32_e64 v0, s12, s16                                // 00000000197C: D52B0000 0000200C
	s_lshl_b64 s[4:5], s[8:9], 2                               // 000000001984: 84848208
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 000000001988: BF8700A9
	s_add_u32 s4, s3, s4                                       // 00000000198C: 80040403
	s_addc_u32 s5, s6, s5                                      // 000000001990: 82050506
	v_fmac_f32_e64 v0, s13, s17                                // 000000001994: D52B0000 0000220D
	s_ashr_i32 s3, s2, 31                                      // 00000000199C: 86039F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000019A0: BF8700A9
	s_lshl_b64 s[2:3], s[2:3], 2                               // 0000000019A4: 84828202
	s_waitcnt lgkmcnt(0)                                       // 0000000019A8: BF89FC07
	v_fmac_f32_e64 v0, s1, s10                                 // 0000000019AC: D52B0000 00001401
	s_add_u32 s2, s4, s2                                       // 0000000019B4: 80020204
	s_addc_u32 s3, s5, s3                                      // 0000000019B8: 82030305
	s_ashr_i32 s1, s0, 31                                      // 0000000019BC: 86019F00
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 0000000019C0: BF8704A1
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 0000000019C4: CA140080 01000080
	s_lshl_b64 s[0:1], s[0:1], 2                               // 0000000019CC: 84808200
	s_add_u32 s0, s2, s0                                       // 0000000019D0: 80000002
	s_addc_u32 s1, s3, s1                                      // 0000000019D4: 82010103
	global_store_b32 v1, v0, s[0:1]                            // 0000000019D8: DC6A0000 00000001
	s_nop 0                                                    // 0000000019E0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000019E4: BFB60003
	s_endpgm                                                   // 0000000019E8: BFB00000
