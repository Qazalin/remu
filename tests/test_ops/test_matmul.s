
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_99_64>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b64 s[4:5], s[0:1], 0x10                            // 000000001704: F4040100 F8000010
	s_load_b128 s[16:19], s[0:1], null                         // 00000000170C: F4080400 F8000000
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s3, s15, 31                                     // 000000001718: 86039F0F
	v_mov_b32_e32 v0, 0                                        // 00000000171C: 7E000280
	s_lshl_b64 s[20:21], s[2:3], 2                             // 000000001720: 84948202
	s_mov_b64 s[22:23], 0                                      // 000000001724: BE960180
	s_waitcnt lgkmcnt(0)                                       // 000000001728: BF89FC07
	s_add_u32 s24, s4, s20                                     // 00000000172C: 80181404
	s_addc_u32 s25, s5, s21                                    // 000000001730: 82191505
	s_add_u32 s18, s18, 60                                     // 000000001734: 8012BC12
	s_addc_u32 s19, s19, 0                                     // 000000001738: 82138013
	s_add_u32 s26, s24, s22                                    // 00000000173C: 801A1618
	s_addc_u32 s27, s25, s23                                   // 000000001740: 821B1719
	s_load_b512 s[0:15], s[18:19], -0x3c                       // 000000001744: F4100009 F81FFFC4
	s_clause 0x7                                               // 00000000174C: BF850007
	s_load_b32 s28, s[26:27], null                             // 000000001750: F400070D F8000000
	s_load_b32 s29, s[26:27], 0x18c                            // 000000001758: F400074D F800018C
	s_load_b32 s30, s[26:27], 0x318                            // 000000001760: F400078D F8000318
	s_load_b32 s31, s[26:27], 0x4a4                            // 000000001768: F40007CD F80004A4
	s_load_b32 s33, s[26:27], 0x630                            // 000000001770: F400084D F8000630
	s_load_b32 s34, s[26:27], 0x7bc                            // 000000001778: F400088D F80007BC
	s_load_b32 s35, s[26:27], 0x948                            // 000000001780: F40008CD F8000948
	s_load_b32 s36, s[26:27], 0xad4                            // 000000001788: F400090D F8000AD4
	s_add_u32 s22, s22, 0x18c0                                 // 000000001790: 8016FF16 000018C0
	s_addc_u32 s23, s23, 0                                     // 000000001798: 82178017
	s_add_u32 s18, s18, 64                                     // 00000000179C: 8012C012
	s_addc_u32 s19, s19, 0                                     // 0000000017A0: 82138013
	s_cmpk_eq_i32 s22, 0x6300                                  // 0000000017A4: B1966300
	s_waitcnt lgkmcnt(0)                                       // 0000000017A8: BF89FC07
	v_fmac_f32_e64 v0, s0, s28                                 // 0000000017AC: D52B0000 00003800
	s_load_b32 s0, s[26:27], 0xc60                             // 0000000017B4: F400000D F8000C60
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017BC: BF8700A1
	v_fmac_f32_e64 v0, s1, s29                                 // 0000000017C0: D52B0000 00003A01
	s_load_b32 s1, s[26:27], 0xdec                             // 0000000017C8: F400004D F8000DEC
	v_fmac_f32_e64 v0, s2, s30                                 // 0000000017D0: D52B0000 00003C02
	s_load_b32 s2, s[26:27], 0xf78                             // 0000000017D8: F400008D F8000F78
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017E0: BF8700A1
	v_fmac_f32_e64 v0, s3, s31                                 // 0000000017E4: D52B0000 00003E03
	s_load_b32 s3, s[26:27], 0x1104                            // 0000000017EC: F40000CD F8001104
	v_fmac_f32_e64 v0, s4, s33                                 // 0000000017F4: D52B0000 00004204
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017FC: BF870091
	v_fmac_f32_e64 v0, s5, s34                                 // 000000001800: D52B0000 00004405
	v_fmac_f32_e64 v0, s6, s35                                 // 000000001808: D52B0000 00004606
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001810: BF870001
	v_fmac_f32_e64 v0, s7, s36                                 // 000000001814: D52B0000 00004807
	s_clause 0x3                                               // 00000000181C: BF850003
	s_load_b32 s4, s[26:27], 0x1290                            // 000000001820: F400010D F8001290
	s_load_b32 s5, s[26:27], 0x141c                            // 000000001828: F400014D F800141C
	s_load_b32 s6, s[26:27], 0x15a8                            // 000000001830: F400018D F80015A8
	s_load_b32 s7, s[26:27], 0x1734                            // 000000001838: F40001CD F8001734
	s_waitcnt lgkmcnt(0)                                       // 000000001840: BF89FC07
	v_fmac_f32_e64 v0, s8, s0                                  // 000000001844: D52B0000 00000008
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000184C: BF870091
	v_fmac_f32_e64 v0, s9, s1                                  // 000000001850: D52B0000 00000209
	v_fmac_f32_e64 v0, s10, s2                                 // 000000001858: D52B0000 0000040A
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001860: BF870091
	v_fmac_f32_e64 v0, s11, s3                                 // 000000001864: D52B0000 0000060B
	v_fmac_f32_e64 v0, s12, s4                                 // 00000000186C: D52B0000 0000080C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001874: BF870091
	v_fmac_f32_e64 v0, s13, s5                                 // 000000001878: D52B0000 00000A0D
	v_fmac_f32_e64 v0, s14, s6                                 // 000000001880: D52B0000 00000C0E
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001888: BF870001
	v_fmac_f32_e64 v0, s15, s7                                 // 00000000188C: D52B0000 00000E0F
	s_cbranch_scc0 65449                                       // 000000001894: BFA1FFA9 <r_99_64+0x3c>
	v_mov_b32_e32 v1, 0                                        // 000000001898: 7E020280
	s_add_u32 s0, s16, s20                                     // 00000000189C: 80001410
	s_addc_u32 s1, s17, s21                                    // 0000000018A0: 82011511
	global_store_b32 v1, v0, s[0:1]                            // 0000000018A4: DC6A0000 00000001
	s_nop 0                                                    // 0000000018AC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018B0: BFB60003
	s_endpgm                                                   // 0000000018B4: BFB00000
