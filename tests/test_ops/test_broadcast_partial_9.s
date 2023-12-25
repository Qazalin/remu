
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001700 <E_5_312_32n2>:
	s_clause 0x1                                               // 000000001700: BF850001
	s_load_b128 s[4:7], s[0:1], null                           // 000000001704: F4080100 F8000000
	s_load_b64 s[0:1], s[0:1], 0x10                            // 00000000170C: F4040000 F8000010
	s_lshl_b32 s2, s14, 5                                      // 000000001714: 8402850E
	s_mulk_i32 s15, 0x2700                                     // 000000001718: B80F2700
	s_add_i32 s2, s2, s13                                      // 00000000171C: 81020D02
	v_mov_b32_e32 v1, 0                                        // 000000001720: 7E020280
	s_add_i32 s2, s2, s15                                      // 000000001724: 81020F02
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001728: BF870499
	s_ashr_i32 s3, s2, 31                                      // 00000000172C: 86039F02
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001730: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001734: BF89FC07
	s_add_u32 s6, s6, s2                                       // 000000001738: 80060206
	s_addc_u32 s7, s7, s3                                      // 00000000173C: 82070307
	s_ashr_i32 s15, s14, 31                                    // 000000001740: 860F9F0E
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(NEXT) | instid1(SALU_CYCLE_1)// 000000001744: BF870499
	s_lshl_b64 s[8:9], s[14:15], 2                             // 000000001748: 8488820E
	s_add_u32 s0, s0, s8                                       // 00000000174C: 80000800
	s_addc_u32 s1, s1, s9                                      // 000000001750: 82010901
	s_load_b32 s6, s[6:7], null                                // 000000001754: F4000183 F8000000
	s_load_b32 s0, s[0:1], null                                // 00000000175C: F4000000 F8000000
	s_waitcnt lgkmcnt(0)                                       // 000000001764: BF89FC07
	v_mul_f32_e64 v0, s6, s0                                   // 000000001768: D5080000 00000006
	s_add_u32 s0, s4, s2                                       // 000000001770: 80000204
	s_addc_u32 s1, s5, s3                                      // 000000001774: 82010305
	global_store_b32 v1, v0, s[0:1]                            // 000000001778: DC6A0000 00000001
	s_nop 0                                                    // 000000001780: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001784: BFB60003
	s_endpgm                                                   // 000000001788: BFB00000
