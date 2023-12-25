
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <r_16_1024_16>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[20:23], s[0:1], null                         // 000000001704: F4080500 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_mov_b32 s2, s15                                          // 000000001714: BE82000F
	s_ashr_i32 s15, s14, 31                                    // 000000001718: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_4) | instid1(SALU_CYCLE_1)// 00000000171C: BF8704D9
	s_lshl_b64 s[24:25], s[14:15], 2                           // 000000001720: 8498820E
	s_waitcnt lgkmcnt(0)                                       // 000000001724: BF89FC07
	s_add_u32 s22, s22, s24                                    // 000000001728: 80161816
	s_addc_u32 s23, s23, s25                                   // 00000000172C: 82171917
	s_lshl_b32 s4, s2, 4                                       // 000000001730: 84048402
	s_ashr_i32 s5, s4, 31                                      // 000000001734: 86059F04
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001738: BF870499
	s_lshl_b64 s[4:5], s[4:5], 2                               // 00000000173C: 84848204
	s_add_u32 s0, s0, s4                                       // 000000001740: 80000400
	s_addc_u32 s1, s1, s5                                      // 000000001744: 82010501
	s_load_b512 s[4:19], s[0:1], null                          // 000000001748: F4100100 F8000000
	s_clause 0x7                                               // 000000001750: BF850007
	s_load_b32 s0, s[22:23], null                              // 000000001754: F400000B F8000000
	s_load_b32 s1, s[22:23], 0x1000                            // 00000000175C: F400004B F8001000
	s_load_b32 s3, s[22:23], 0x2000                            // 000000001764: F40000CB F8002000
	s_load_b32 s26, s[22:23], 0x3000                           // 00000000176C: F400068B F8003000
	s_load_b32 s27, s[22:23], 0x4000                           // 000000001774: F40006CB F8004000
	s_load_b32 s28, s[22:23], 0x5000                           // 00000000177C: F400070B F8005000
	s_load_b32 s29, s[22:23], 0x6000                           // 000000001784: F400074B F8006000
	s_load_b32 s30, s[22:23], 0x7000                           // 00000000178C: F400078B F8007000
	s_waitcnt lgkmcnt(0)                                       // 000000001794: BF89FC07
	v_fma_f32 v0, s0, s4, 0                                    // 000000001798: D6130000 02000800
	s_load_b32 s0, s[22:23], 0x8000                            // 0000000017A0: F400000B F8008000
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 0000000017A8: BF8700A1
	v_fmac_f32_e64 v0, s1, s5                                  // 0000000017AC: D52B0000 00000A01
	s_load_b32 s1, s[22:23], 0x9000                            // 0000000017B4: F400004B F8009000
	v_fmac_f32_e64 v0, s3, s6                                  // 0000000017BC: D52B0000 00000C03
	s_load_b32 s3, s[22:23], 0xa000                            // 0000000017C4: F40000CB F800A000
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000017CC: BF870091
	v_fmac_f32_e64 v0, s26, s7                                 // 0000000017D0: D52B0000 00000E1A
	v_fmac_f32_e64 v0, s27, s8                                 // 0000000017D8: D52B0000 0000101B
	s_clause 0x4                                               // 0000000017E0: BF850004
	s_load_b32 s4, s[22:23], 0xb000                            // 0000000017E4: F400010B F800B000
	s_load_b32 s5, s[22:23], 0xc000                            // 0000000017EC: F400014B F800C000
	s_load_b32 s6, s[22:23], 0xd000                            // 0000000017F4: F400018B F800D000
	s_load_b32 s7, s[22:23], 0xe000                            // 0000000017FC: F40001CB F800E000
	s_load_b32 s8, s[22:23], 0xf000                            // 000000001804: F400020B F800F000
	v_fmac_f32_e64 v0, s28, s9                                 // 00000000180C: D52B0000 0000121C
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001814: BF870091
	v_fmac_f32_e64 v0, s29, s10                                // 000000001818: D52B0000 0000141D
	v_fmac_f32_e64 v0, s30, s11                                // 000000001820: D52B0000 0000161E
	s_waitcnt lgkmcnt(0)                                       // 000000001828: BF89FC07
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_1) | instid1(VALU_DEP_1)// 00000000182C: BF8700A1
	v_fmac_f32_e64 v0, s0, s12                                 // 000000001830: D52B0000 00001800
	s_lshl_b32 s0, s2, 10                                      // 000000001838: 84008A02
	v_fmac_f32_e64 v0, s1, s13                                 // 00000000183C: D52B0000 00001A01
	s_ashr_i32 s1, s0, 31                                      // 000000001844: 86019F00
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001848: BF870099
	s_lshl_b64 s[0:1], s[0:1], 2                               // 00000000184C: 84808200
	v_fmac_f32_e64 v0, s3, s14                                 // 000000001850: D52B0000 00001C03
	s_add_u32 s0, s20, s0                                      // 000000001858: 80000014
	s_addc_u32 s1, s21, s1                                     // 00000000185C: 82010115
	s_add_u32 s0, s0, s24                                      // 000000001860: 80001800
	s_addc_u32 s1, s1, s25                                     // 000000001864: 82011901
	v_fmac_f32_e64 v0, s4, s15                                 // 000000001868: D52B0000 00001E04
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001870: BF870091
	v_fmac_f32_e64 v0, s5, s16                                 // 000000001874: D52B0000 00002005
	v_fmac_f32_e64 v0, s6, s17                                 // 00000000187C: D52B0000 00002206
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001884: BF870091
	v_fmac_f32_e64 v0, s7, s18                                 // 000000001888: D52B0000 00002407
	v_fmac_f32_e64 v0, s8, s19                                 // 000000001890: D52B0000 00002608
	s_delay_alu instid0(VALU_DEP_1)                            // 000000001898: BF870001
	v_dual_mov_b32 v1, 0 :: v_dual_max_f32 v0, 0, v0           // 00000000189C: CA140080 01000080
	global_store_b32 v1, v0, s[0:1]                            // 0000000018A4: DC6A0000 00000001
	s_nop 0                                                    // 0000000018AC: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000018B0: BFB60003
	s_endpgm                                                   // 0000000018B4: BFB00000
